#!/bin/bash

VENV_PATH="./.venv"

source "$VENV_PATH/Scripts/activate" #Windows command
#source "$VENV_PATH/bin/activate" #Linux command

cd NewsScrapper || exit
scrapy crawl news

deactivate
