 # 安装依赖
    cd deepseek-tui-py
    pip install -r requirements.txt
    
    # CLI 模式 (需要 API key)
    export DEEPSEEK_API_KEY=sk
    python -m deepseek_tui.cli exec "Hello, what can you do?"
    
    # 配置
    python -m deepseek_tui.cli config set model deepseek-v4-flash
    python -m deepseek_tui.cli config list
    
    # 运行测试
    python -m pytest tests/ -v
    
    # TUI 模式 (需 Textual)
    python -m deepseek_tui.cli
    
    # 或
    python -c "from deepseek_tui.tui import DeepSeekApp; DeepSeekApp().run()"