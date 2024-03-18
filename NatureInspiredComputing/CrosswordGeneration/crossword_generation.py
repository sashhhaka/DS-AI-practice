"""
    ---------------------------
    Evolutionary algorithm crossword generator
    Author: Alexandra Vabnits
"""

import random
import time

"""
    ---------------------------
    Initialization of crosswords
"""


def init_crossword(words):
    """
    Initialize the crossword puzzle with the given words.
    :param words: a list of words
    :return: a list of tuples (word, position, direction)
    """
    _words = []
    for word in words:
        direction = random.randint(0, 1)
        if direction == 0:
            position = (random.randint(0, 20 - len(word)), random.randint(0, 19))

        else:
            position = (random.randint(0, 19), random.randint(0, 20 - len(word)))
        _words.append((word, position, direction))
    return _words


def init_population(words, size):
    """
    Initialize the population with the given words.
    :param words: a list of words
    :param size: the size of the population
    :return: a list of crosswords
    """
    population = []
    for _ in range(size):
        population.append(init_crossword(words))
    return population


def parse_words(file="input1.txt"):
    """
    Parses a list of words from given file
    """
    words = []
    file = "inputs/" + file
    with open(file, "r") as f:
        for line in f:
            words.append(line.strip())
    return words


def to_table(crossword):
    """
    Translates a list of tuples to a grid
    """
    table = [['.' for _ in range(20)] for _ in range(20)]
    for word in crossword:
        letters, (x, y), direction = word
        if direction == 0:
            for i, letter in enumerate(letters):
                table[y][x + i] = letter
        else:
            for i, letter in enumerate(letters):
                table[y + i][x] = letter
    return table


def to_string(table):
    """
    Method for printing a grid as a string
    """
    buff = ""
    for row in table:
        buff += " ".join(row) + "\n"
    return buff


def print_table(table):
    """
    Prints table
    """
    buff = ""
    for row in table:
        buff += " ".join(row) + "\n"
    print(buff)


def print_table_to_file(table, filename):
    """
    Writes the table to file
    """


"""
    ---------------------------
    Helper functions
"""


def cells_around(position):
    """
    By given position, returns a list of cells around it
    :param position: a tuple (x, y)
    :return: a list of tuples (x, y)
    """
    x, y = position
    return [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]


def word_cells(letters, position, direction):
    """
    By given word, returns a list of cells that the word occupies
    :param letters: characters of the word
    :param position: a tuple (x, y)
    :param direction: 0 - horizontal, 1 - vertical
    :return: a list of tuples (letter, (x, y))
    """
    x, y = position
    return [(letters[i], (x + i * (not direction), y + i * direction)) for i in range(len(letters))]


"""
    ---------------------------
    Fitness function
"""


def intersecting_words(crossword):
    """
    Returns a graph of intersecting words
    :param crossword: a list of tuples (word, position, direction)
    :return: a graph represented as a list of lists
    """
    characters = {}
    directions = {}
    indices = {}
    graph = [[] for _ in range(len(crossword))]

    for i, word in enumerate(crossword):
        # letters, (x, y), direction = word
        for char, pos in word_cells(*word):
            if pos in characters and characters[pos] == char and directions[pos] != word[2]:
                graph[i].append(indices[pos])
                graph[indices[pos]].append(i)

            characters[pos] = char
            directions[pos] = word[2]
            indices[pos] = i

    return graph


def get_components(crossword):
    """
    Returns a list of components of the crossword
    :param crossword: a list of tuples (word, position, direction)
    :return: a list of lists of tuples (word, position, direction)
    """

    def find(component, parents):
        if parents[component] == -1:
            return component
        return find(parents[component], parents)

    def union(component1, component2, parents, ranks):
        root1 = find(component1, parents)
        root2 = find(component2, parents)

        if root1 != root2:
            if ranks[root1] > ranks[root2]:
                parents[root2] = root1
            elif ranks[root1] < ranks[root2]:
                parents[root1] = root2
            else:
                parents[root2] = root1
                ranks[root1] += 1

    graph = intersecting_words(crossword)
    num_nodes = len(graph)
    components = [0] * num_nodes
    current_component = 0

    parents = [-1] * num_nodes
    ranks = [0] * num_nodes

    for word in range(num_nodes):
        for neighbor in graph[word]:
            root_word = find(word, parents)
            root_neighbor = find(neighbor, parents)

            if root_word != root_neighbor:
                union(root_word, root_neighbor, parents, ranks)

    component_mapping = {}
    for word in range(num_nodes):
        root = find(word, parents)
        if root not in component_mapping:
            current_component += 1
            component_mapping[root] = current_component

        components[word] = component_mapping[root]

    components_words = [[] for _ in range(max(components))]
    for i, component in enumerate(components):
        components_words[component - 1].append(crossword[i])

    return components_words


def conflicts_and_components(crossword):
    """
    Returns the number of conflicts in letters of the crossword
    and the number of connected components
    :param crossword: a list of tuples (word, position, direction)
    :return: a tuple (number of conflicts, number of connected components)
    """

    def find(component, parents):
        """
        Find the root of a component
        """
        if parents[component] == -1:
            return component
        return find(parents[component], parents)

    def union(component1, component2, parents, ranks):
        """
        Unite two components
        """
        root1 = find(component1, parents)
        root2 = find(component2, parents)

        if root1 != root2:
            if ranks[root1] > ranks[root2]:
                parents[root2] = root1
            elif ranks[root1] < ranks[root2]:
                parents[root1] = root2
            else:
                parents[root2] = root1
                ranks[root1] += 1

    def update_values(cells, current_index):
        """
        Update the values of the cells
        """
        for char, pos in cells:
            indexes[pos] = current_index
            directions[pos] = word[2]
            characters[pos] = char

    def check_around(cells, current_index):
        """
        Check for conflicts around the cells and exclude the intersections from the conflicts
        """
        intersections = []
        conflicts = [0] * len(crossword)

        for char, pos in cells:
            for cell in cells_around(pos):
                if cell in indexes:
                    conflicts[indexes[cell]] += 1

            if pos in characters and characters[pos] == char and directions[pos] != word[2]:
                union(current_index, indexes[pos], parents, ranks)
                intersections.append(indexes[pos])

        conflicts[current_index] = 0
        for j in intersections:
            conflicts[j] = 0

        return sum(conflicts)

    characters = {}
    indexes = {}
    directions = {}
    summed_conflicts = 0
    current_index = 0

    # Union-Find related structures
    parents = [-1] * len(crossword)
    ranks = [0] * len(crossword)

    for word in crossword:
        cells = word_cells(*word)
        conflicts = check_around(cells, current_index)
        summed_conflicts += conflicts
        update_values(cells, current_index)
        current_index += 1

    # Count connected components using the union-find structure
    connected_components = sum(1 for parent in parents if parent == -1)

    return summed_conflicts, connected_components


def fitness(crossword):
    """
    Main fitness function
    :param crossword: a list of tuples (word, position, direction)
    :return: a float value
    """
    value = 1.0
    max_words_component = max(len(component) for component in get_components(crossword))
    n_conflicts, n_components = conflicts_and_components(crossword)

    weights = [
        0.8,
        0.9,
    ]
    parameters = [
        n_components - 1,
        n_conflicts,
    ]
    for parameter, weight in zip(parameters, weights):
        value *= weight ** parameter
    value *= max_words_component / len(crossword)
    return value


"""
    ---------------------------
    Mutation and crossover
"""


def mutate_word(word):
    """
    Mutation helper function
    :param word: a tuple (word, position, direction)
    :return: a tuple (word, position, direction)
    """
    letters, (_, _), direction = word
    new_direction = random.randint(0, 1)
    if new_direction == 0:
        new_position = (random.randint(0, 20 - len(letters)), random.randint(0, 19))
    else:
        new_position = (random.randint(0, 19), random.randint(0, 20 - len(letters)))
    return letters, new_position, new_direction


def mutation(crossword, mutation_rate=1.0):
    """
    Mutates the crossword with the given mutation rate
    :param crossword: a list of tuples (word, position, direction)
    :param mutation_rate: a probability of mutation for each word
    :return: a list of tuples (word, position, direction)
    """
    mutant = []
    for word in crossword:
        if random.random() < mutation_rate:
            mutant.append(mutate_word(word))
        else:
            mutant.append(word)
    return mutant


def cross(word1, word2):
    """
    Crossover helper function
    :param word1: a tuple (word, position, direction)
    :param word2: a tuple (word, position, direction)
    :return: a tuple (word, position, direction)
    """
    return (word1[0], word1[1], word1[2]) if random.randint(0, 1) == 0 else (word2[0], word2[1], word2[2])


def crossover(crossword1, crossword2):
    """
    Crossover function
    :param crossword1: a list of tuples (word, position, direction)
    :param crossword2: a list of tuples (word, position, direction)
    :return: a list of tuples (word, position, direction)
    """
    crossword1.sort(key=lambda x: x[0])
    crossword2.sort(key=lambda x: x[0])
    child = list(crossword1)

    i = 0
    for word1, word2 in zip(crossword1, crossword2):
        child[i] = cross(word1, word2)
        i += 1
    return child


def translate(component, dx, dy):
    """
    Translation helper function
    :param component: a list of tuples (word, position, direction)
    :param dx: a horizontal translation
    :param dy: a vertical translation
    :return: a list of tuples (word, position, direction)
    """
    translated = []
    for word, (x, y), direction in component:
        translated.append((word, (x + dx, y + dy), direction))
    return translated


def translation(crossword):
    """
    Translation function
    Translates a crossword component by a random value and possibly rotates a 1-word component
    :param crossword: a list of tuples (word, position, direction)
    :return: a list of tuples (word, position, direction)
    """
    components = get_components(crossword)  # refactor
    new_crossword = []
    for component in components:
        # find a list of words to be translated
        if len(component) == 1:
            new_crossword.append(mutate_word(component[0]))

        else:
            # translate the words
            # find valid positions for the words
            min_x = min(x for _, (x, y), _ in component)
            min_y = min(y for _, (x, y), _ in component)
            max_h_x = max(x + len(word) for word, (x, y), direction in component if direction == 0)
            max_v_x = max(x for word, (x, y), direction in component if direction == 1)
            max_h_y = max(y for word, (x, y), direction in component if direction == 0)
            max_v_y = max(y + len(word) for word, (x, y), direction in component if direction == 1)
            max_x = max(max_h_x, max_v_x)
            max_y = max(max_h_y, max_v_y)

            d_x = random.randint(-min_x, 20 - max_x)
            d_y = random.randint(-min_y, 20 - max_y)
            # translate the words in a component
            new_crossword.extend(translate(component, d_x, d_y))
    return new_crossword


"""
    ---------------------------
    Evolutionary algorithm
"""


def fitness_check(fitness_array, duplicates=20):
    """
    Checks if the last 20 fitness values are the same
    :param fitness_array:
    :param duplicates:
    :return: True if the last 20 fitness values are the same, False otherwise
    """
    flag = True
    fitness_array = fitness_array[-duplicates:]
    for i in range(len(fitness_array) - 1):
        if fitness_array[i] != fitness_array[i + 1]:
            flag = False
            break
    return flag


def check_for_duplicates(crossword):
    """
    Checks if there are duplicate words in the crossword
    :param crossword: a tuple (word, position, direction)
    :return: True if there are duplicate words, False otherwise
    """
    words = []
    for word in crossword:
        words.append(word[0])
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if words[i] == words[j]:
                return True
    return False


def evolution(words, mutation_rate=0.6, population_size=300):
    """
    Main evolutionary algorithm function
    :param words: a list of words
    :param mutation_rate: a probability of mutation for each word
    :param population_size: the size of the population
    :return: a list of tuples (word, position, direction), execution time
    """
    best_individuals_buffer = []
    start = time.time()
    population = init_population(words, population_size)

    fitness_changes = []
    fitness_prev = 0
    i = 0
    while True:
        population = sorted(population, key=lambda x: fitness(x), reverse=True)
        if fitness_check(fitness_changes, 200):
            translated = []
            for crossword in population:
                trans = translation(crossword)
                translated.append(trans)

            population = []
            for crossword in translated:
                population.append(crossword)
            population = sorted(population, key=lambda x: fitness(x), reverse=True)

        # if there are <= 3 components and last 20 best fitness value are the same, translate them
        if len(get_components(population[0])) <= 3 and fitness_check(fitness_changes, 20):
            translated = []
            for crossword in population:
                trans = translation(crossword)
                translated.append(trans)

            population = []
            for crossword in translated:
                population.append(crossword)
            population = sorted(population, key=lambda x: fitness(x), reverse=True)

        fitness_prev = fitness(population[0])
        fitness_changes.append(fitness_prev)
        if fitness(population[0]) == 1.0:
            return population[0], time.time() - start

        if time.time() - start > 300:
            best_individuals_buffer.append(population[0])
            best_individuals = sorted(best_individuals_buffer, key=lambda x: fitness(x), reverse=True)
            return best_individuals[0], time.time() - start
        if time.time() - start > 100:
            best_individuals_buffer.append(population[0])
            population = init_population(words, population_size)
            population = sorted(population, key=lambda x: fitness(x), reverse=True)
        distribution = {
            "best": 90,
            "worst": 10,
            "legacy": 100,
        }
        # main evolution step
        best = population[:distribution["best"]]
        # save 5% of the worst population
        worst = population[-distribution["worst"]:]
        population = best + worst
        new_population = []
        for _ in range(distribution["legacy"]):
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            new_population.append(child)
            child = mutation(child, mutation_rate)
            new_population.append(child)
        population = population + new_population
        i += 1


words_sample = [
    "cage",
    "cemetery",
    "chemistry",
    "engine",
    "fairytale",
    "gate",
    "pillow",
    "train",
    "widow",
    "wine"
]

POPULATION_SIZE = 300


def main():
    import os
    dir = "inputs/"
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    files.sort(key=lambda x: int(x[5:-4]))
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    for i, file in enumerate(files):
        words = parse_words(file)
        best, execution_time = evolution(words, population_size=POPULATION_SIZE)
        with open("outputs/output" + str(i + 1) + ".txt", "w") as f:

            best.sort(key=lambda x: words.index(x[0]))
            for word in best:
                f.write(str(word[1][0]) + " " + str(word[1][1]) + " " + str(word[2]) + "\n")
            table = to_table(best)
            f.write("\n")
            f.write(to_string(table))
            print("File", i + 1, "done", execution_time, len(words), fitness(best))


def test():
    words = parse_words("input1.txt")
    best, execution_time = evolution(words, population_size=POPULATION_SIZE)
    print("Execution time:", execution_time)
    print("Fitness:", fitness(best))
    print_table(to_table(best))


if __name__ == "__main__":
    main()
    # test()
