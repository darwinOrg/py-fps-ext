import os
import pathlib
from typing import Optional

import fitz
import magic_pdf.model as model_config
from loguru import logger
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from pdfminer.high_level import extract_text

model_config.__use_inside_model__ = True
output_txt_filename = 'cv.txt'
output_md_filename = 'cv.md'


def extract_pdf_text_by_magic_pdf(pdf_path: str, output_dir: str, word_count_min: int, max_page: int,
                                  parse_method: Optional[str]) -> str:
    pdf_path = truncate_pdf_over_pages(pdf_path, output_dir, max_page)
    extracted_text = extract_text(pdf_path).strip()
    if len(extracted_text) < word_count_min:
        return ''

    name_without_suffix = os.path.basename(pdf_path)
    local_image_dir = output_dir + "/images"
    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(output_dir)
    image_dir = os.path.basename(local_image_dir)

    pdf_bytes = FileBasedDataReader("").read(pdf_path)

    ds = PymuDocDataset(pdf_bytes)

    if parse_method is None or parse_method == '' or parse_method == 'auto':
        classify = ds.classify()
    else:
        classify = parse_method

    if classify == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    ### draw model result on each page
    infer_result.draw_model(os.path.join(output_dir, f"{name_without_suffix}_model.pdf"))

    ### draw layout result on each page
    pipe_result.draw_layout(os.path.join(output_dir, f"{name_without_suffix}_layout.pdf"))

    ### draw spans result on each page
    pipe_result.draw_span(os.path.join(output_dir, f"{name_without_suffix}_spans.pdf"))

    ### dump markdown
    pipe_result.dump_md(md_writer, output_md_filename, image_dir)

    ### dump content list
    pipe_result.dump_content_list(md_writer, f"{name_without_suffix}_content_list.json", image_dir)

    md_file_path = os.path.join(output_dir, output_md_filename)
    with open(md_file_path, 'r') as md_file:
        lines = md_file.readlines()

    if lines and lines[0].startswith('![](images'):
        image_path = lines[0].split('(')[1].split(')')[0]
        logo_dir = os.path.join(output_dir, 'logo')
        os.makedirs(logo_dir, exist_ok=True)
        new_image_path = os.path.join(logo_dir, os.path.basename(image_path))
        os.rename(os.path.join(output_dir, image_path), new_image_path)

        lines[0] = lines[0].replace('images', 'logo')
        with open(md_file_path, 'w') as md_file:
            md_file.writelines(lines)

    valid_word_count = 0
    for line in lines:
        if line.strip() == '' or line.strip().startswith('![]'):
            continue
        valid_word_count += len(line.strip())
    if valid_word_count < word_count_min:
        return ''

    logger.info(f'extract_pdf_text_by_magic_pdf({pdf_path} -> {md_file_path}) success')
    return md_file_path


def truncate_pdf_over_pages(input_pdf_path: str, output_dir: str, max_page: int) -> str:
    with fitz.open(input_pdf_path) as doc:
        total_pages = len(doc)
        if total_pages <= max_page:
            return input_pdf_path

        with fitz.open() as new_doc:
            for page_num in range(min(max_page, total_pages)):
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            output_pdf_path = str(
                pathlib.Path(output_dir).joinpath(pathlib.Path(input_pdf_path).stem + "_truncated.pdf"))
            new_doc.save(output_pdf_path)
            print(f"Truncated document to {max_page} pages and saved as {output_pdf_path}.")
            return output_pdf_path
