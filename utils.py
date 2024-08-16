import re
from xdsl.parser import Parser
from xdsl.context import MLContext
from xdsl.dialects import get_all_dialects

def create_context_with_all_dialects():
    context = MLContext()
    for dialect_name, dialect_factory in get_all_dialects().items():
        context.register_dialect(dialect_name, dialect_factory)
    return context

def remove_comments(content):
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    return content

def parse_mlir_file(file_path, context):
    with open(file_path, 'r') as f:
        content = f.read()

    content = remove_comments(content)

    parser = Parser(context, content)
    module = parser.parse_module()
    return module