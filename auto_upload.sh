#!/bin/bash

rm -f ./dict/*
python3 -m build
python3 -m twine ./dict/*