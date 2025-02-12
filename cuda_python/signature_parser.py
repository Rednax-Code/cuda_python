"""
This holds a function that extracts the input and output datatypes from a string representation.
"""

from re import match

string_to_type = {
	"int": int,
	"float": float,
	"list[int]": list[int],
	"list[float]": list[float]
}

def parse(signature: str):
	# Use regex to extract inputs and output
	re_match = match(r"\(\((.*?)\),\s*\((.*?)\)\)", signature)
	input_str, output_str = re_match.groups()
	inputs = input_str.split(', ')
	outputs = output_str.split(', ')
	input_types = [string_to_type[i] for i in inputs]
	output_types = [string_to_type[i] for i in outputs]
	return input_types, output_types