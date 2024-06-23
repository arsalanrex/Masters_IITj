import sys

# Misra-Gries algorithm for k-frequent element detection
def misra_gries(k, stream):
    # Initialize an empty dictionary
    freq_dict = {}

    # Process each character in the stream
    for char in stream:
        if char in freq_dict:
            # If the character is already in the dictionary, increment its count
            freq_dict[char] += 1
        elif len(freq_dict) < k - 1:
            # If there are fewer than k-1 elements in the dictionary, add the new character
            freq_dict[char] = 1
        else:
            # Decrement count of every element in the dictionary
            keys_to_remove = [] # List to keep track of elements to remove
            for key in freq_dict:
                freq_dict[key] -= 1 # Decrement each element's count by 1
                if freq_dict[key] == 0:
                    # If any count drops to zero, add that element to the removal list
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del freq_dict[key] # Remove elements with a count of zero from the dictionary

    # Final output: elements with their approximate counts
    return freq_dict

if __name__ == "__main__":

    k = int(sys.argv[1])  # First argument: k
    stream = sys.argv[2]  # Second argument: data stream

    # Ensure k is at least 2 (since k-1 must be at least 1)
    if k < 2:
        print("k must be at least 2 parameters long")
        sys.exit(1)

    # Run the Misra-Gries algorithm and get the result
    result = misra_gries(k, stream)

    # Print the result as a dictionary
    print(result)
