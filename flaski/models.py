# coding: utf-8
from sqlalchemy import Column, Integer, String, Boolean
from flaski.database import Base
from datetime import datetime

class Image(Base):
    __tablename__ = 'images'
    id          = Column(Integer, primary_key=True)
    filename    = Column(String(128), unique=True)
    label       = Column(Integer)
    is_complete = Column(Boolean)

    def __init__(self, filename=None, label=None, is_complete=False):
        self.filename    = filename
        self.label       = label
        self.is_complete = is_complete
