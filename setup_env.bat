@echo off
echo Creazione dell'ambiente Conda astraFBP...
call conda env create -f environment.yml
echo Attivazione ambiente...
call conda activate astraFBP
echo Installazione ambiente completata
pause
