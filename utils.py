from xdsl.context import MLContext
from xdsl.dialects import get_all_dialects

def create_context_with_all_dialects():
    context = MLContext()
    for dialect_name, dialect_factory in get_all_dialects().items():
        context.register_dialect(dialect_name, dialect_factory)
    return context
