def term_vec_to_file(terms, vector, filepath):
    """
    Write a word vector to a filepath, in a human readable format.
    Includes the words themselves.
    """
    with open(filepath, 'w') as file:
        for i, token in enumerate(terms):
            file.write(f'{token}: {vector[i]}\n')