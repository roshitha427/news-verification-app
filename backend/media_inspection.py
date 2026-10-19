"""Small, optional Gemini vision pass for media description and exact visible text."""
import base64
import io
import json
import logging
import re

import requests
_SCHEMA = {
    'type': 'OBJECT',
    'properties': {
        'description': {'type': 'STRING'},
        'visible_text': {'type': 'STRING'},
        'visual_artifacts': {'type': 'STRING'},
        'visual_type': {'type': 'STRING', 'enum': [
            'photograph', 'cartoon_or_illustration', 'screenshot_or_document', 'other', 'unclear'
        ]},
    },
    'required': ['description', 'visible_text', 'visual_artifacts', 'visual_type'],
}


def _image_part(image):
    """Encode a bounded JPEG so uploads do not create large vision requests."""
    copy = image.convert('RGB')
    copy.thumbnail((768, 768))
    output = io.BytesIO()
    copy.save(output, format='JPEG', quality=78, optimize=True)
    return {'inline_data': {'mime_type': 'image/jpeg',
                            'data': base64.b64encode(output.getvalue()).decode('ascii')}}


def inspect_media(images, api_key, model='gemini-3.5-flash-lite'):
    """Read visible text first, then describe the image or sampled video frames.

    This is OCR and scene-description assistance, not an authenticity or
    fact-check verdict. Video inspection receives the sampled frames from the
    analyzer, up to its configured twelve-frame limit.
    """
    if not api_key:
        return {'status': 'not_configured', 'description': '', 'visible_text': '',
                'visual_artifacts': '', 'visual_type': 'unclear', 'ocr_performed': False}
    if not images:
        return {'status': 'unavailable', 'description': '', 'visible_text': '',
                'visual_artifacts': '', 'visual_type': 'unclear', 'ocr_performed': False}
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}', model):
        return {'status': 'invalid_model_config', 'description': '', 'visible_text': '',
                'visual_artifacts': '', 'visual_type': 'unclear', 'ocr_performed': False}

    indexes = list(range(min(12, len(images))))
    prompt = (
        'Do the tasks in this order. First, act as OCR: transcribe every readable word, number, date, label, '
        'headline, sign, or subtitle shown in the attached media, preserving wording and order. For video frames, '
        'combine related text and do not duplicate the same overlay. Put the transcription in visible_text; use an '
        'empty string only when no readable text is present. Do not invent or complete cropped words. Second, give '
        'a concise description of the scene and notable subjects or objects. Name a recognizable fictional character '
        'only when clear. Then note visible irregularities a reviewer could inspect, such as inconsistent lighting, '
        'warped edges, repeated textures, or malformed details. Put only observable features in visual_artifacts; '
        'do not call them proof or infer the cause. Set visual_type to photograph, cartoon_or_illustration, '
        'screenshot_or_document, other, or unclear. Do not decide whether the media is real or AI-generated; a '
        'separate image model does that. Keep description to one short sentence, visual_artifacts to at most 500 '
        'characters, and visible_text to at most 2000 characters. Return only the requested JSON fields.'
    )
    parts = [{'text': prompt}]
    try:
        parts.extend(_image_part(images[index]) for index in indexes)
        response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
            headers={'x-goog-api-key': api_key, 'Content-Type': 'application/json'},
            json={'contents': [{'parts': parts}],
                  'generationConfig': {'maxOutputTokens': 1200,
                      'responseFormat': {'text': {'mimeType': 'APPLICATION_JSON', 'schema': _SCHEMA}}}},
            timeout=(4, 20),
        )
        response.raise_for_status()
        payload = response.json()
        raw = ''.join(part.get('text', '') for part in
                      payload['candidates'][0]['content']['parts'] if isinstance(part, dict)).strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError('invalid response')
        description = str(parsed.get('description') or '').strip()[:400]
        visible_text = str(parsed.get('visible_text') or '').strip()[:2000]
        visual_artifacts = str(parsed.get('visual_artifacts') or '').strip()[:500]
        visual_type = str(parsed.get('visual_type') or 'unclear')
        if visual_type not in {'photograph', 'cartoon_or_illustration', 'screenshot_or_document', 'other', 'unclear'}:
            visual_type = 'unclear'
        if not description and not visible_text:
            raise ValueError('empty response')
        return {'status': 'completed', 'description': description,
                'visible_text': visible_text, 'visual_artifacts': visual_artifacts,
                'visual_type': visual_type,
                'sampled_images': len(indexes), 'ocr_performed': True}
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError, OSError) as exc:
        logging.getLogger(__name__).warning('Gemini media inspection did not complete (%s).',
                                           exc.__class__.__name__)
        return {'status': 'unavailable', 'description': '', 'visible_text': '',
                'visual_artifacts': '', 'visual_type': 'unclear', 'ocr_performed': False}
