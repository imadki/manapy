
import ast

# Define a custom AST visitor class
class FunctionNameVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function_names = []

    def visit_FunctionDef(self, node):
        # Extract the function name and add it to the list of function names
        self.function_names.append(node.name)
        self.generic_visit(node)

# Function to parse function names from a Python file
def parse_function_names(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
        # Parse the code into an AST
        tree = ast.parse(code)
        # Create an instance of the FunctionNameVisitor
        visitor = FunctionNameVisitor()
        # Visit the AST to extract function names
        visitor.visit(tree)
        # Return the list of function names
        return visitor.function_names