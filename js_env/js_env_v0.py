from .env.env import JSEnv

def create_js_v0(**kwargs):
    """
    Initialize and return an instance of CustomEnvironment version 0.
    Additional arguments can be passed to configure the environment.
    """
    
    # Create an instance of the environment with the specific configurations
    env = JSEnv(**kwargs)
    return env
