# -*- coding: utf-8 -*-
"""Probe the real APIs of the installed SOTA packages."""
import importlib
import inspect
import warnings
warnings.filterwarnings('ignore')
# --- SpaGCN ---
from SpaGCN import SpaGCN as SGC
print('SpaGCN methods:',
      [(n, str(inspect.signature(getattr(SGC, n))))
       for n in ['train', 'predict', 'set_l']])
# --- SpaceFlow ---
SF = importlib.import_module('SpaceFlow')
print('\nSpaceFlow names:', [n for n in dir(SF) if not n.startswith('_')])
sf_cls = getattr(SF, 'SpaceFlow', None)
if sf_cls:
    print('init:', inspect.signature(sf_cls.__init__))
    print('methods:', [(n, str(inspect.signature(getattr(sf_cls, n))))
                       for n in dir(sf_cls)
                       if not n.startswith('_') and callable(getattr(sf_cls, n))])
# --- GraphST ---
from GraphST import GraphST as GST
print('\nGraphST init:', inspect.signature(GST.__init__))
print('methods:', [(n, str(inspect.signature(getattr(GST, n))))
                   for n in dir(GST)
                   if not n.startswith('_') and callable(getattr(GST, n))])
import GraphST
print('\nGraphST module names:',
      [n for n in dir(GraphST) if not n.startswith('_')])
