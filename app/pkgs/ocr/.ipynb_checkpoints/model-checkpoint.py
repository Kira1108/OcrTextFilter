from __future__ import annotations
from typing import Any
from cnstd import CnStd
from cnocr import CnOcr
import cv2
from abc import ABC, abstractmethod
import logging

std = CnStd()
cn_ocr = CnOcr()
logger = logging.getLogger(__name__)

class Reader(ABC):
    
    @abstractmethod
    def read(self,Any):
        pass
    
class Processor(ABC):
    @abstractmethod
    def process(self, Any):
        pass
    
class Recgonizer(ABC):
    @abstractmethod
    def ocr(self, Any):
        pass

class ImageReader(Reader):
    def read(self,path):
        logger.info(f'read Image from {path}')
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

class TextLocalizer(Processor):
    def _detect(self, img):
        return img,std.detect(img)
    
    def _crop(self, img, boxes):
        logger.info(f"find {len(boxes)} text boxes")
        imgs = []
        for i,box in enumerate(boxes):
            b = box['box']
            xmin, xmax = b[:,0].min(),b[:,0].max()
            ymin, ymax = b[:,1].min(),b[:,1].max()
            sub_img = img[ymin:ymax, xmin:xmax]
            imgs.append(sub_img)
        return imgs
    
    def process(self,img):
        return self._crop(*self._detect(img)) 


class OcrSingleLine(Recgonizer):
    def ocr(self, img):
        return cn_ocr.ocr_for_single_line(img)
    
    
class OcrCrops(Recgonizer):
    def ocr(self, imgs):
        logger.info("ocr texts on all crops")
        return [''.join(cn_ocr.ocr_for_single_line(img)) for img in imgs]
    
    
class ImageToText():
    ''' 
        reader = ImageReader()
        localizer = TextLocalizer()
        recgonizer = OcrCrops()
    '''
    
    def __init__(self, reader = None, localizer = None, recgonizer = None):
        self.reader = reader if reader else ImageReader()
        self.localizer = localizer if localizer else TextLocalizer()
        self.recgonizer = recgonizer if recgonizer else OcrCrops()
    
    def detect(self, img):
        crops = self.localizer.process(self.reader.read(img))
        return self.recgonizer.ocr(crops)