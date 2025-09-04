'''Program that will take in a base and height
and print out the area of a triangle with that
base and height. '''

# Get user's triangle base and height
triangle_base = input("Enter in your triangle's base: ")
triangle_height = input("Enter in your triangle's height:")

# Convert to floating point
triangle_base = float(triangle_base)
triangle_height = float(triangle_height)

# Calculate the area
triangle_area = 0.5 * triangle_base * triangle_height

# Print the result
print("Your triangle's area is:\t", triangle_area)