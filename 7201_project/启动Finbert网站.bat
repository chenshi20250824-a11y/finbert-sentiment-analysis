@echo off
color 0A
cd /d "C:\Users\Ast\Desktop\7201_project第二版\7201_project"
call conda activate CISC7201
start http://localhost:5000
python app_flask. py