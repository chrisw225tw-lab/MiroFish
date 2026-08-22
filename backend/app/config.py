"""\n配置管理\n统一从项目根目录的 .env 文件加载配置\n"""

import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


class Config:
    """Flask配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # JSON配置 - 禁用ASCII转义，让中文直接显示
    JSON_AS_ASCII = False
    
    # LLM配置（统一使用OpenAI格式）
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    # LLM 模型 failover 链（逗号分隔；主模型失败时按顺序切换，同 base_url/key）
    LLM_MODEL_FALLBACKS = [
        m.strip()
        for m in os.environ.get('LLM_MODEL_FALLBACKS', '').split(',')
        if m.strip()
    ]

    # ===== 图后端配置 =====
    # cloud: Zep Cloud（默认，上游行为）
    # graphiti: 本地自建 Graphiti（graphiti-core 直连 FalkorDB）
    ZEP_BACKEND = os.environ.get('ZEP_BACKEND', 'cloud').strip().lower()

    # Graphiti 本地配置（仅 ZEP_BACKEND=graphiti 时需要）
    GRAPHITI_FALKORDB_URI = os.environ.get('GRAPHITI_FALKORDB_URI', 'redis://127.0.0.1:6380')
    # FalkorDB 图键（graph key）。与 graphiti-mcp 的 `main` 分开，避免索引/数据互相干扰。
    GRAPHITI_FALKORDB_DATABASE = os.environ.get('GRAPHITI_FALKORDB_DATABASE', 'mirofish')
    # Graphiti 抽取用 LLM（缺省回落到 LLM_* 配置）
    GRAPHITI_LLM_API_KEY = os.environ.get('GRAPHITI_LLM_API_KEY')
    GRAPHITI_LLM_BASE_URL = os.environ.get('GRAPHITI_LLM_BASE_URL')
    GRAPHITI_LLM_MODEL_NAME = os.environ.get('GRAPHITI_LLM_MODEL_NAME')
    GRAPHITI_LLM_MODEL_FALLBACKS = [
        m.strip()
        for m in os.environ.get('GRAPHITI_LLM_MODEL_FALLBACKS', '').split(',')
        if m.strip()
    ]
    # Graphiti Embedder（主：GPU 服务器；fallback：本地 ollama）
    GRAPHITI_EMBEDDER_API_KEY = os.environ.get('GRAPHITI_EMBEDDER_API_KEY', 'ollama')
    GRAPHITI_EMBEDDER_BASE_URL = os.environ.get('GRAPHITI_EMBEDDER_BASE_URL')
    GRAPHITI_EMBEDDER_FALLBACK_BASE_URL = os.environ.get('GRAPHITI_EMBEDDER_FALLBACK_BASE_URL')
    GRAPHITI_EMBEDDER_MODEL = os.environ.get('GRAPHITI_EMBEDDER_MODEL', 'nomic-embed-text')
    GRAPHITI_EMBEDDER_DIM = int(os.environ.get('GRAPHITI_EMBEDDER_DIM', '768'))
    
    # Zep Cloud 配置
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小
    
    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls) -> list[str]:
        """验证必要配置"""
        errors: list[str] = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY 未配置")
        if cls.ZEP_BACKEND == 'cloud':
            if not cls.ZEP_API_KEY:
                errors.append("ZEP_API_KEY 未配置（ZEP_BACKEND=cloud 时必需）")
            if os.environ.get("ZEP_API_URL"):
                errors.append("ZEP_API_URL 不受支持；MiroFish 仅连接 Zep Cloud")
        elif cls.ZEP_BACKEND == 'graphiti':
            if not cls.GRAPHITI_FALKORDB_URI:
                errors.append("GRAPHITI_FALKORDB_URI 未配置")
            if not cls.GRAPHITI_EMBEDDER_BASE_URL:
                errors.append("GRAPHITI_EMBEDDER_BASE_URL 未配置（ZEP_BACKEND=graphiti 时必需）")
            graphiti_llm_key = cls.GRAPHITI_LLM_API_KEY or cls.LLM_API_KEY
            if not graphiti_llm_key:
                errors.append("GRAPHITI_LLM_API_KEY 或 LLM_API_KEY 未配置")
        else:
            errors.append(f"ZEP_BACKEND 取值无效: {cls.ZEP_BACKEND}（应为 cloud 或 graphiti）")
        if cls.DEBUG:
            import warnings
            warnings.warn("Flask DEBUG mode is enabled. Do not use in production.", RuntimeWarning)
        return errors
