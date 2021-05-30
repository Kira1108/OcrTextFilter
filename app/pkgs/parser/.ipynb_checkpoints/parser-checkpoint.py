from typing import List
from ...config import config
from ..ocr import ImageToText
from ..texts import DFAFilter

class OCRTextParser():
    def __init__(self, image_handler = None, text_handler = None):
        self.image_handler = image_handler if image_handler else ImageToText()
        self.text_handler = text_handler if text_handler else DFAFilter()
        self._initialize()
        
    def _initialize(self):
        self.text_handler.parse(config['sensative_path'])
        
    def parse(self, img_path):
        texts = self.image_handler.detect(img_path)
        return {t:self.text_handler.is_contain_sensi_key_word(t) 
                for t in texts}
    
    
class TextParser():
    def __init__(self, text_handler = None):
        self.text_handler = text_handler if text_handler else DFAFilter()
        self._initialize()
        
    def _initialize(self):
        self.text_handler.parse(config['sensative_path'])
        
    def parse(self, texts:List[str]):
        return {t:self.text_handler.is_contain_sensi_key_word(t) for t in texts}
        
    