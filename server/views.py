import os
import json
import csv
import re
import unicodedata
import sys
import typing as tp
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import pandas as pd
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, Adafactor
from transformers import get_constant_schedule_with_warmup
from django.core.files.base import ContentFile
from sacremoses import MosesPunctNormalizer

def initialize_mpn(lang):
    mpn = MosesPunctNormalizer(lang=lang)
    mpn.substitutions = [
        (re.compile(r), sub) for r, sub in mpn.substitutions
    ]
    return mpn

def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)
    print("Normalization finished")
    return replace_non_printing_char

replace_nonprint = get_non_printing_char_replacer(" ")

def preproc(text, mpn):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    clean = unicodedata.normalize("NFKC", clean)
    return clean

def read_csv_file(file_path):
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    delimiters = [',', '\t']
    
    for encoding in encodings:
        for delimiter in delimiters:
            try:
                with open(file_path, 'r', encoding=encoding) as csvfile:
                    start = csvfile.read(1024)
                    csvfile.seek(0)
                    dialect = csv.Sniffer().sniff(start)
                    dialect.delimiter = delimiter
                    
                    df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, quoting=csv.QUOTE_MINIMAL, dialect=dialect)
                
                if df.shape[1] >= 2:
                    print(f"Successfully read CSV with encoding: {encoding} and delimiter: {repr(delimiter)}")
                    return df
                else:
                    print(f"File read successfully but doesn't have at least two columns. Trying next format...")
            except Exception as e:
                print(f"Failed to read with encoding: {encoding} and delimiter: {repr(delimiter)}. Error: {str(e)}")
                continue
    
    raise ValueError("Failed to read the CSV file with all attempted encodings and delimiters.")

@csrf_exempt
def train_nmt(request):
    if request.method == 'POST':
        source_lang = request.POST.get('sourceLang')
        target_lang = request.POST.get('targetLang')
        path = request.POST.get('path')
        mosesPunctNormalizer = request.POST.get('MosesPunctNormalizer')
        batch_size = int(request.POST.get('batchSize', 32))
        max_length = int(request.POST.get('maxLength', 128))
        warmup_steps = int(request.POST.get('warmupSteps', 500))
        training_steps = int(request.POST.get('trainingSteps', 10000))
        learning_rate = float(request.POST.get('learningRate', 1e-4))
        weight_decay = float(request.POST.get('weightDecay', 0.01))
        
        required_params = ['sourceLang', 'targetLang', 'MosesPunctNormalizer']
        missing_params = [param for param in required_params if param not in request.POST]
        if missing_params:
            return JsonResponse({'error': f'Missing required parameters: {", ".join(missing_params)}'}, status=400)

        if 'inputFile' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        uploaded_file = request.FILES['inputFile']
        file_name = default_storage.save('temp.csv', ContentFile(uploaded_file.read()))
        file_path = default_storage.path(file_name)

        try:
            df = read_csv_file(file_path)
            if df.shape[1] < 2:
                raise ValueError("CSV file must have at least two columns")
            df.columns = ['source', 'target'] + list(df.columns[2:])
        except Exception as e:
            default_storage.delete(file_name)
            return JsonResponse({'error': f'Failed to read CSV file: {str(e)}'}, status=400)
        
        default_storage.delete(file_name)
        
        mpn = initialize_mpn(mosesPunctNormalizer)
        df['source'] = df['source'].apply(lambda x: preproc(x, mpn))
        df['target'] = df['target'].apply(lambda x: preproc(x, mpn))

        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

        fix_tokenizer(tokenizer, new_lang=source_lang)
        model.resize_token_embeddings(len(tokenizer))
        print("Going to optimizer now")
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=False,
            relative_step=False,
            lr=learning_rate,
            clip_threshold=1.0,
            weight_decay=weight_decay,
        )
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        model.train()
        for step in range(training_steps):
            source_texts, target_texts = get_batch(df, batch_size)
            tokenizer.src_lang = source_lang
            inputs = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            tokenizer.tgt_lang = target_lang
            labels = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

            outputs = model(**inputs, labels=labels.input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")

        if not os.path.exists(f"models/{path}"):
            os.makedirs(f"models/{path}")
        model.save_pretrained(f"models/{path}")
        tokenizer.save_pretrained(f"models/{path}")

        return JsonResponse({'message': 'Training completed successfully'})

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def fix_tokenizer(new_lang):
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len - 1
    tokenizer.id_to_lang_code[old_len - 1] = new_lang
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset
    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}
    print("Tokenizer fixed")

def get_batch(df, batch_size):
    batch = df.sample(batch_size)
    source_texts = batch.iloc[:, 0].tolist() 
    target_texts = batch.iloc[:, 1].tolist()
    print("Get batch finished") 
    return source_texts, target_texts

def tokenizer_for_translation(tokenizer, new_lang):
    print("Starting tokenizer_for_translation function")
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len - 1
    tokenizer.id_to_lang_code[old_len - 1] = new_lang
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset
    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}
    print("Finished tokenizer_for_translation function")

def translate2(text, model, tokenizer, src_lang='spa_Latn', tgt_lang='eng_Latn', max_input_length=1024, a=32, b=3, num_beams=4, **kwargs):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    print("Going to inputs now")
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams,
        **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)

@csrf_exempt
def translation_endpoint(request):
    print("Translation endpoint called")
    try:
        data = json.loads(request.body)
        text = data.get('text')
        src_lang = data.get('src_lang')
        tgt_lang = data.get('tgt_lang')
        model_path = data.get('path')
        max_length = data.get('max_length', 128)

        print(f"Received request: text={text}, src_lang={src_lang}, tgt_lang={tgt_lang}, max_length={max_length}")

        if not all([text, src_lang, tgt_lang, model_path]):
            print("Missing required parameters")
            return JsonResponse({'error': 'Missing required parameters'}, status=400)
        
        if not os.path.exists(f"models/{model_path}"):
            print("Model path does not exist")
            return JsonResponse({'error': 'Model path does not exist'}, status=400)
        
        print("Loading model and tokenizer")
        model = AutoModelForSeq2SeqLM.from_pretrained(f"models/{model_path}")
        tokenizer = NllbTokenizer.from_pretrained(f"models/{model_path}")
        
        print("Preparing tokenizer for translation")
        tokenizer_for_translation(tokenizer, src_lang)
        
        print("Starting translation")
        translated_text = translate2(
            text,
            model,
            tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_input_length=max_length
        )

        print("Translation completed")
        return JsonResponse({'translated_text': translated_text})

    except json.JSONDecodeError:
        print("Invalid JSON received")
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
