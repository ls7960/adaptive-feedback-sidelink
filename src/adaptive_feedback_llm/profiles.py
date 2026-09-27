"""The sole paper profile; aliases retain the public workflow imports."""

from adaptive_feedback_llm.config import LLMConfig
from adaptive_feedback_llm.solvers.llm.provider import ProviderSettings

PROMPT_VERSION = "1.0"
PROMPT_SHA256 = "d35e6ad71900f775134f95707566448a85274defb5ae9f07fa096e919cacfad2"

ThinkingConfig = LLMConfig
ThinkingSettings = ProviderSettings
