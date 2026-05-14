@echo off
echo 正在启动T5后端服务...
start cmd /k "CALL conda.bat activate glm && python backend\t5_server.py"

timeout /t 2

echo 正在启动BART后端服务...
start cmd /k "CALL conda.bat activate T5Bot1 && python backend\bart_server.py"

timeout /t 2

echo 正在启动前端聊天界面...
start cmd /k "CALL conda.bat activate chatapp && python frontend/chat_new_beautiful3.py"


echo 所有程序已启动！
pause
