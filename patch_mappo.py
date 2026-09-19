import re

with open('marl/algorithms/mappo.py', 'r') as f:
    content = f.read()

old_config = """        actor_config = self.config.get('actor_config', {})
        critic_config = self.config.get('critic_config', {})"""

new_config = """        actor_config = self.config.get('actor', self.config.get('actor_config', {}))
        critic_config = self.config.get('critic', self.config.get('critic_config', {}))"""

content = content.replace(old_config, new_config)

with open('marl/algorithms/mappo.py', 'w') as f:
    f.write(content)
