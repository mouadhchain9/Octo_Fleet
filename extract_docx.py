import zipfile
import xml.etree.ElementTree as ET
import sys

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.XML(xml_content)
            
            NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            
            text = []
            for node in tree.iter(f'{NAMESPACE}t'):
                if node.text:
                    text.append(node.text)
                    
            return '\n'.join(text)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open('thesis_annex.md', 'w', encoding='utf-8') as f:
            f.write(extract_text_from_docx(sys.argv[1]))
    else:
        print("Provide docx path")
