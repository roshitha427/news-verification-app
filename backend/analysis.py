import hashlib
import logging
import math
import re
import threading
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from backend.verification import verify_submission
from ml.vision import VisualModel
from ml.text_cleaning import clean_text, content_language
from backend.speech import SpeechTranscriber
from backend.generation import GenerationDetector
from backend.report_summary import filter_relevant_evidence, summarize_report
from backend.local_evidence import evidence_ready
from backend.media_inspection import inspect_media
from ml.temporal import TemporalModel
from ml.ai_text import AITextModel

Image.MAX_IMAGE_PIXELS = 30_000_000


def _is_visual_caption_only(text):
    """Treat short labels such as 'Shinchan cartoon picture' as descriptions."""
    words = re.findall(r"[\w'-]+", (text or '').lower(), flags=re.UNICODE)
    visual_terms = {'picture', 'photo', 'image', 'cartoon', 'drawing', 'illustration',
                    'character', 'screenshot', 'poster', 'meme', 'scene', 'person',
                    'man', 'woman', 'child', 'dog', 'cat', 'landscape', 'building'}
    claim_terms = {'won', 'lost', 'died', 'killed', 'elected', 'arrested', 'announced',
                   'approved', 'banned', 'launched', 'caused', 'increased', 'decreased',
                   'confirmed', 'reported', 'claimed', 'reveals', 'proves'}
    has_number = any(char.isdigit() for char in text or '')
    visual_caption = len(words) <= 10 and bool(set(words) & visual_terms)
    return visual_caption and not bool(set(words) & claim_terms) and not has_number


def _usable_media_claim(text):
    words = re.findall(r"[\w'-]+", (text or ''), flags=re.UNICODE)
    return len((text or '').strip()) >= 10 and len(words) >= 2


class Analyzer:
    def __init__(self, config):
        self.config = config
        self.visual = VisualModel(config['IMAGE_MODEL_PATH'])
        self.text_pipe = None
        self.text_lock = threading.Lock()
        self.speech = SpeechTranscriber(config.get('WHISPER_MODEL'))
        self.generation = GenerationDetector(config.get('AI_IMAGE_MODEL_PATH'))
        self.target_accuracy = config.get('MODEL_TARGET_ACCURACY', .90)
        self.temporal = TemporalModel(config.get('TEMPORAL_MODEL_PATH'))
        self.ai_text = AITextModel(config['AI_TEXT_MODEL_PATH']) if config.get('AI_TEXT_MODEL_PATH') else None

    def readiness(self):
        text_path = Path(self.config['TEXT_MODEL_PATH'])
        provider = self.config.get('EVIDENCE_PROVIDER', 'local')
        return {'image': self.visual.ready(), 'video': self.visual.ready() or self.temporal.ready(),
                'temporal': self.temporal.ready(),
                'text': (text_path / 'config.json').is_file() and (text_path / 'model-card.json').is_file()
                    and any((text_path / filename).is_file() for filename in ('model.safetensors', 'pytorch_model.bin')),
                'speech': self.speech.ready(),
                'ai_generation': self.generation.ready(),
                'ai_writing': bool(self.ai_text and self.ai_text.ready),
                'gemini_report_summary': bool(self.config.get('GEMINI_API_KEY')),
                'fact_check': bool(self.config.get('GOOGLE_FACT_CHECK_API_KEY')) and provider in ('google', 'auto'),
                'evidence_search': (provider == 'local' and evidence_ready(self.config.get('EVIDENCE_MODEL_PATH'))) or provider in ('public', 'rss', 'gdelt', 'auto') or (provider == 'google' and bool(self.config.get('GOOGLE_FACT_CHECK_API_KEY')))}

    def text(self, text):
        if not self.readiness()['text']:
            return None, {'name': 'Transformer text classifier', 'version': 'not installed', 'status': 'not_configured'}
        import json
        from transformers import pipeline
        directory = Path(self.config['TEXT_MODEL_PATH'])
        card = json.loads((directory / 'model-card.json').read_text(encoding='utf8'))
        if content_language(text) not in card.get('languages', ['en']):
            return None, {'name': card.get('name', 'BERT'), 'version': card.get('version', '1'),
                          'status': 'unsupported_language', 'evaluation': card.get('evaluation')}
        word_count = len(clean_text(text).split())
        if word_count < 20 or (text.strip().endswith('?') and word_count < 80):
            return None, {'name':card.get('name', 'BERT text classifier'), 'version':card.get('version', '1'),
                          'status':'claim_out_of_domain', 'evaluation':card.get('evaluation'),
                          'external_evaluations':card.get('external_evaluations', {})}
        if card.get('fake_label') is None:
            raise ValueError('The text model must declare its fake_label in model-card.json.')
        with self.text_lock:
            if self.text_pipe is None:
                self.text_pipe = pipeline('text-classification', model=str(directory), tokenizer=str(directory), device=-1)
            scores = self.text_pipe(clean_text(text), truncation=True, max_length=card.get('max_length', 512), top_k=None)
        fake = next((x['score'] for x in scores if x['label'] == card['fake_label']), None)
        if fake is not None:
            odds = math.log(max(1e-7, fake) / max(1e-7, 1 - fake))
            fake = 1 / (1 + math.exp(-odds / card.get('temperature', 1.)))
        evaluation = card.get('evaluation')
        return fake, {'name': card.get('name', 'BERT text classifier'), 'version': card.get('version', '1'),
                      'status': 'ready', 'evaluation': evaluation,
                      'requested_target_accuracy': self.target_accuracy,
                      'requested_target_met': bool(evaluation and evaluation.get('accuracy', 0) >= self.target_accuracy),
                      'external_evaluations': card.get('external_evaluations', {})}

    def analyze(self, kind, text, filepath, language, sources):
        supplied_claim = (text or '').strip()
        result = {'fake_percentage': None, 'real_percentage': None, 'ai_percentage': None, 'confidence': None,
                  'models': [], 'observations': [], 'limitations': [
                    'AI generation is separate from factual accuracy. A generated illustration does not establish whether the surrounding news claim is true.',
                    'Model confidence is not measured accuracy and does not verify a news claim.'],
                  'scope': 'No model prediction is available for this content.'}
        probability = None
        text_model_card = None
        if kind == 'text':
            self._assess_ai_writing(text, result)
            try:
                probability, card = self.text(text)
            except Exception:
                logging.getLogger(__name__).exception('Text classifier failed; continuing with evidence analysis.')
                probability, card = None, {'name': 'Transformer text classifier', 'version': 'unknown',
                                           'status': 'unavailable'}
            text_model_card = card
            result['models'].append(card)
            self._text_domain_warning(result, card)
            result['scope'] = 'Text fake/real classification based on the installed training corpus. Independent evidence is required.'
            if card['status'] == 'unsupported_language':
                result['limitations'].append('This text model was evaluated on English only. It abstains for Hindi, Telugu, and other unsupported scripts.')
                result['metric_note'] = 'The text classifier was evaluated on English only, so it did not calculate fake/real scores or confidence for this language.'
            elif card['status'] == 'claim_out_of_domain':
                result['scope'] = 'The news-article classifier abstained on this short claim or question. Review the concise evidence assessment and source details in the downloaded PDF.'
                result['limitations'].append('The text classifier was trained on full news articles and performs poorly on short political claims. It abstains here; use the retrieved evidence details in the downloaded PDF to assess the claim.')
                metric_note = (
                    'The text classifier abstained because this is a short claim outside its full-article training domain. '
                    'Fake/real scores and confidence were not computed.'
                )
                ai_status = (result.get('ai_assessment') or {}).get('status')
                if ai_status == 'too_short':
                    metric_note += f" The separate AI-writing detector needs at least {result['ai_assessment'].get('min_words', 80)} English words."
                elif ai_status == 'unsupported_language':
                    metric_note += ' The separate AI-writing detector has not been evaluated for this language.'
                elif not self.ai_text or not self.ai_text.ready:
                    metric_note += ' The separate AI-writing detector is not configured.'
                result['metric_note'] = metric_note
            elif probability is None:
                result['limitations'].append('Text model not trained: supply a labeled text dataset to enable transformer classification.')
                result['metric_note'] = 'The text classifier is not configured; fake/real scores and confidence were not computed.'
        else:
            images = []
            timestamps = []
            if kind == 'image':
                with Image.open(filepath) as image:
                    image.load()
                    img = ImageOps.exif_transpose(image).convert('RGB')
                    result['observations'].append({'label': 'Dimensions', 'value': f'{img.width} × {img.height} px'})
                    images = [img.copy()]
            else:
                import cv2
                cap = cv2.VideoCapture(str(filepath))
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or frames <= 0:
                        raise ValueError('This video could not be decoded. Use an MP4 or WebM file.')
                    duration = frames / fps
                    if not math.isfinite(duration) or duration > 120:
                        raise ValueError('Please use a video of two minutes or less.')
                    samples = min(12, max(1, int(frames))) if self.temporal.ready() else min(12, max(1, int(duration)))
                    for index in np.rint(np.linspace(0, frames-1, samples)).astype(int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                        ok, frame = cap.read()
                        if ok:
                            if frame.shape[0] * frame.shape[1] > 30_000_000:
                                raise ValueError('Video frame dimensions exceed the supported limit.')
                            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                            timestamps.append(round(float(index / fps), 2))
                    if not images:
                        raise ValueError('No readable frames were found in the video.')
                    result['observations'].extend([{'label': 'Duration', 'value': f'{duration:.1f} seconds'},
                                                   {'label': 'Sampled frames', 'value': str(len(images))}])
                finally:
                    cap.release()
                if not self.temporal.ready():
                    result['limitations'].append('The AI-origin estimate uses sampled video frames; motion and audio authenticity are not assessed by that detector.')
            frame_probabilities, frame_model_version = None, None
            media_inspection = inspect_media(
                images, self.config.get('GEMINI_API_KEY'),
                self.config.get('GEMINI_MODEL', 'gemini-3.5-flash-lite'))
            result['media_inspection'] = media_inspection
            if self.visual.ready():
                probabilities, card = self.visual.predict(images)
                frame_probabilities, frame_model_version = probabilities, card.get('version')
                probability = float(np.mean(probabilities))
                result['models'].append(dict(card, status='ready'))
                if card.get('evaluation'):
                    result['models'][-1].update(requested_target_accuracy=self.target_accuracy,
                        requested_target_met=card['evaluation']['accuracy'] >= self.target_accuracy)
                result['limitations'].extend(card.get('limitations', []))
                result['limitations'].append('The visual classifier was trained on face-image crops. Performance on general news photos or non-face video frames is not established.')
                if card.get('evaluation', {}).get('accuracy', 0) < self.target_accuracy:
                    result['limitations'].append(f'The visual classifier did not meet the requested {self.target_accuracy * 100:.0f}% held-out accuracy target. Treat its scores as experimental.')
                if kind == 'video':
                    result['frames'] = [{'seconds': sec} for sec in timestamps]
                result['scope'] = card.get('scope', 'Dataset-specific fake/real visual classification. Does not determine whether the surrounding news is true.')
            else:
                result['models'].append({'name': 'Shared visual classifier', 'version': 'not trained', 'status': 'not_configured'})
                result['limitations'].append('No trained visual artifact is installed. Media was inspected but authenticity was not assessed.')
                if kind != 'video' or not self.temporal.ready():
                    result['metric_note'] = 'The visual classifier is not configured, so fake/real scores and model confidence were not computed.'
            if kind == 'video' and self.temporal.ready():
                if len(images) < 3:
                    result['limitations'].append('Too few readable frames for the learned temporal model. The video score uses available visual frames only.')
                else:
                    try:
                        probability, temporal_card = self.temporal.predict(images, frame_probabilities, frame_model_version)
                        temporal_card.update(status='ready', requested_target_accuracy=self.target_accuracy,
                            requested_target_met=temporal_card['evaluation']['accuracy'] >= self.target_accuracy)
                        result['models'].append(temporal_card)
                        result['temporal_assessment'] = {'status': 'ready', 'sampled_frames': len(images), 'primary_score': True}
                        result['scope'] = temporal_card['scope']
                        result['limitations'].extend(temporal_card.get('limitations', []))
                        result['limitations'].append('A separate learned consistency model also checked sampled frames. The AI-origin estimate remains separate; audio authenticity is not assessed.')
                    except (OSError, ValueError, RuntimeError, ImportError):
                        result['temporal_assessment'] = {'status': 'unavailable'}
                        result['limitations'].append('The learned video model could not run. Available frame-based analysis is retained.')
            visual_type = media_inspection.get('visual_type')
            # Run the AI-origin detector on every visual type. The prior skip for
            # cartoons and screenshots left those uploads without the requested
            # simple result; the report still marks these scores experimental.
            if self.generation.ready():
                try:
                    ai_scores, ai_card = self.generation.predict(images)
                    result['ai_percentage'] = round(float(np.mean(ai_scores)) * 100, 1)
                    result['ai_assessment'] = {'status': 'experimental', 'calibrated': False,
                        'scope': 'sampled_video_frames' if kind == 'video' else 'image',
                        'sample_count': len(ai_scores), 'source_url': ai_card['source_url']}
                    result['models'].append(dict(ai_card, status='ready'))
                    result['limitations'].extend(ai_card.get('limitations', [])[:2])
                    if kind == 'video':
                        result['limitations'].extend(ai_card.get('limitations', [])[2:])
                    if kind == 'video':
                        result['frames'] = [{'seconds': sec, 'ai_percentage': round(score * 100, 1)}
                                            for sec, score in zip(timestamps, ai_scores)]
                    if visual_type in ('cartoon_or_illustration', 'screenshot_or_document'):
                        result['limitations'].append(
                            f'The AI-origin estimate for {visual_type.replace("_", " ")} is experimental and has not been independently validated for this media type.')
                except (OSError, RuntimeError, ValueError, ImportError):
                    result['ai_assessment'] = {'status': 'unavailable'}
                    result['limitations'].append('The AI-image detector could not run. Its percentage was left unassessed; the other analysis results are still available.')
            else:
                result['ai_assessment'] = {'status': 'not_configured', 'calibrated': False,
                    'scope': 'sampled_video_frames' if kind == 'video' else 'image'}
                result['limitations'].append('No dedicated AI-image detector is installed. The AI-generated percentage is not assessed for this media.')
            # Face detection provides an observation, not a deepfake verdict.
            import cv2
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_counts = []
            for image in images:
                gray = cv2.cvtColor(np.array(image.resize((min(640, image.width), min(640, image.height)))), cv2.COLOR_RGB2GRAY)
                face_counts.append(len(cascade.detectMultiScale(gray, 1.1, 5)))
            result['observations'].append({'label': 'Faces detected (maximum per frame)', 'value': str(max(face_counts, default=0))})
            if kind == 'video' and self.readiness()['speech']:
                try:
                    speech = self.speech.transcribe(filepath)
                    transcript = speech.pop('text')
                    result['speech'] = speech
                    result['models'].append({key: speech[key] for key in ('name', 'version', 'status')})
                    if transcript.strip():
                        result['transcript'] = transcript
                        text = '\n'.join([text, transcript]).strip()
                        speech_probability, speech_card = self.text(transcript)
                        self._text_domain_warning(result, speech_card)
                        result['speech_text_assessment'] = {'fake_percentage': None if speech_probability is None else round(speech_probability * 100, 1), 'model': speech_card}
                        result['limitations'].append('Speech is an automatic transcript that may contain errors. Its text assessment is separate from frame scores; review the words against the recording.')
                    elif speech['status'] not in ('no_audio', 'silence', 'no_speech_detected'):
                        result['limitations'].append('Video audio could not be transcribed; visual analysis and any supplied claim were still assessed.')
                except (OSError, RuntimeError, ValueError):
                    result['speech'] = {'name': 'Whisper tiny speech transcription', 'status': 'unavailable'}
                    result['limitations'].append('Speech transcription was configured but could not run. Check local Whisper files and FFmpeg installation.')
        if kind == 'text' and probability is not None:
            result.update(fake_percentage=round(probability * 100, 1), real_percentage=round((1 - probability) * 100, 1),
                          confidence=round(max(probability, 1 - probability) * 100, 1))
        if kind == 'text':
            result['text_model_assessment'] = self._text_model_assessment(probability, text_model_card or {})
        else:
            # For media, the headline pair represents the AI-origin model only.
            # Its output is not a calibrated confidence and says nothing about
            # whether an accompanying news claim is true.
            ai_score = result.get('ai_percentage')
            result['fake_percentage'] = ai_score
            result['real_percentage'] = None if ai_score is None else round(100 - ai_score, 1)
            result['confidence'] = None
            result['metric_note'] = ('The model estimates AI generation from visual patterns. This score is experimental, not calibrated accuracy or proof of origin.'
                                     if ai_score is not None else
                                     'The AI-origin model could not assess this upload, so no real/AI estimate is shown.')

            inspection = result.get('media_inspection') or {}
            extracted_text = str(inspection.get('visible_text') or '').strip()
            result['ocr_text'] = extracted_text
            claim_parts = []
            if supplied_claim and not _is_visual_caption_only(supplied_claim) and _usable_media_claim(supplied_claim):
                claim_parts.append(supplied_claim)
            # Keep the exact OCR visible in the report, but only search text
            # long enough to carry a useful proposition or query.
            if _usable_media_claim(extracted_text):
                claim_parts.append(extracted_text)
            transcript = str(result.get('transcript') or '').strip()
            if _usable_media_claim(transcript):
                claim_parts.append(transcript)
            claim = '\n'.join(dict.fromkeys(part for part in claim_parts if part))[:12000]

            visual_type = inspection.get('visual_type', 'unclear')
            media_verdict = ('ai_generated' if ai_score >= 50 else 'likely_real') if isinstance(ai_score, (int, float)) else 'classification_unavailable'
            result['media_origin'] = {
                'status': (result.get('ai_assessment') or {}).get('status', 'not_assessed'),
                'verdict': media_verdict, 'score': ai_score, 'visual_type': visual_type,
            }
            result['media_content_mode'] = ('claim_review' if claim else
                                            'text_detected' if extracted_text or transcript else 'visual_only')
            result['claim_text'] = claim
            if not claim:
                description = str(inspection.get('description') or '').strip()
                visual_artifacts = str(inspection.get('visual_artifacts') or '').strip()
                if ai_score is None:
                    origin_label = 'Image origin could not be classified.'
                elif ai_score >= 50:
                    origin_label = 'The model classifies this as likely AI-generated.'
                else:
                    origin_label = 'The model classifies this as likely real (not flagged as AI-generated).'
                summary_parts = [origin_label]
                if description:
                    summary_parts.append(description)
                if visual_artifacts:
                    summary_parts.append('Visible details to review: ' + visual_artifacts)
                if extracted_text:
                    summary_parts.append('OCR text: ' + extracted_text[:1000])
                if transcript:
                    summary_parts.append('Speech transcript: ' + transcript[:1000])
                if (extracted_text or transcript) and not claim:
                    summary_parts.append('Text was detected but did not contain enough context for a reliable source search.')
                result['verification'] = {'status': 'not_applicable', 'provider_status': 'not_run', 'evidence': []}
                result['report_summary'] = {
                    'schema_version': 5, 'mode': 'media_only',
                    'headline': ('Likely AI-generated' if ai_score is not None and ai_score >= 50 else
                                 'Likely real' if ai_score is not None else 'Origin classification unavailable'),
                    'summary': ' '.join(summary_parts),
                    'reason': result['metric_note'],
                }

        claim = result.get('claim_text') if kind in ('image', 'video') else text
        if kind == 'text' or claim:
            detected = content_language(claim)
            verification = verify_submission(claim, detected if detected in ('en', 'hi', 'te') else language,
                                  self.config.get('GOOGLE_FACT_CHECK_API_KEY'), sources,
                                  provider=self.config.get('EVIDENCE_PROVIDER', 'local'),
                                  evidence_path=self.config.get('EVIDENCE_MODEL_PATH'))
            checked_at = verification.get('checked_at')
            for item in verification.get('evidence') or []:
                if not isinstance(item, dict):
                    continue
                item.setdefault('retrieved_at', checked_at)
                item.setdefault('source_type', item.get('kind', 'source'))
            verification = filter_relevant_evidence(claim, verification)
            result['verification'] = verification
            result['report_summary'] = summarize_report(
                claim, kind, result, verification,
                api_key=self.config.get('GEMINI_API_KEY'),
                model=self.config.get('GEMINI_MODEL', 'gemini-3.5-flash-lite'),
            )
        if filepath:
            with open(filepath, 'rb') as media:
                result['sha256'] = hashlib.file_digest(media, 'sha256').hexdigest()
        return result

    @staticmethod
    def _text_domain_warning(result, card):
        diagnostic = card.get('external_evaluations', {}).get('liar_strict_binary')
        if diagnostic and card.get('status') == 'ready':
            result['text_domain_warning'] = {'benchmark': 'LIAR short political claims',
                'accuracy': diagnostic['accuracy'], 'test_count': diagnostic['test_count'],
                'article_accuracy': (card.get('evaluation') or {}).get('accuracy'),
                'source_url': diagnostic['source_url']}
            note = (f"External claim benchmark: this article-trained model scored {diagnostic['accuracy'] * 100:.1f}% on "
                    f"{diagnostic['test_count']} unambiguous LIAR political claims. Its historical article accuracy does not establish claim verification accuracy.")
            if note not in result['limitations']:
                result['limitations'].append(note)

    @staticmethod
    def _text_model_assessment(probability, card):
        status = card.get('status', 'not_configured')
        model = card.get('name', 'Transformer text classifier')
        if probability is None:
            reasons = {
                'not_configured': 'No trained text classifier is installed. Gemini evidence comparison can still assess the claim when its API is configured.',
                'claim_out_of_domain': 'The classifier abstained because this short claim or question is outside its full-article training domain.',
                'unsupported_language': 'The classifier abstained because this language is outside its evaluated languages.',
                'unavailable': 'The text classifier could not complete this run. The Gemini evidence assessment is shown separately.',
            }
            return {'status': status, 'model': model, 'version': card.get('version'),
                    'verdict': 'not_assessed', 'reason': reasons.get(status,
                        'The text classifier did not produce a prediction for this input.'),
                    'evaluation': card.get('evaluation')}

        fake = round(probability * 100, 1)
        real = round((1 - probability) * 100, 1)
        confidence = round(max(probability, 1 - probability) * 100, 1)
        favored = 'fake/manipulated' if probability >= 0.5 else 'real/authentic'
        return {
            'status': 'assessed', 'model': model, 'version': card.get('version'),
            'verdict': 'fake' if probability >= 0.5 else 'authentic',
            'fake_percentage': fake, 'real_percentage': real, 'confidence': confidence,
            'reason': (f'The classifier favored the {favored} class with {confidence:.1f}% model confidence. '
                       'This is a learned text-pattern signal, not a fact-check of the claim.'),
            'evaluation': card.get('evaluation'),
        }

    def _assess_ai_writing(self, text, result):
        if not self.ai_text or not self.ai_text.ready:
            result['limitations'].append('No validated AI-writing detector is installed. The AI-generated percentage is not assessed for text.')
            return
        try:
            probability, card = self.ai_text.predict(text)
            if probability is None:
                result['ai_assessment'] = {'status': card['status'], 'scope': 'text', 'min_words': card.get('min_words',80)}
                if card['status'] == 'too_short':
                    result['limitations'].append(f"The AI-writing model requires at least {card.get('min_words',80)} English words. Short-text authorship is not assessed.")
                elif card['status'] == 'unsupported_language':
                    result['limitations'].append('The AI-writing model has not been evaluated for this language. Authorship is not assessed.')
                return
            result['ai_percentage'] = round(probability*100,1)
            result['ai_assessment'] = {'status': 'experimental', 'scope': 'text', 'calibrated': True,
                'sample_count': 1, 'source_url': card.get('dataset',{}).get('source_url', 'https://huggingface.co/datasets/Hello-SimpleAI/HC3'),
                'benchmark_accuracy': card.get('evaluation',{}).get('accuracy'),
                'domain_accuracy': card.get('domain_evaluation',{}).get('wiki_domain_holdout',{}).get('accuracy')}
            result['models'].append(dict(card, status='ready', purpose='ai_generation', score_scope='text',
                source_url=result['ai_assessment']['source_url']))
            result['limitations'].extend(card.get('limitations', []))
        except (OSError, ValueError, RuntimeError, ImportError, KeyError):
            result['ai_assessment'] = {'status': 'unavailable', 'scope': 'text'}
            result['limitations'].append('The AI-writing detector could not run. Its percentage was left unassessed; news classification and evidence retrieval are separate.')
