### PrEvelOp
### Synthetic data generation for demonstration and testing
### Author: Kaspar Bunne

import numpy as np
import pandas as pd


def generate_toy_dataset(n_samples=300, random_state=42):
    """
    Generate a synthetic manufacturing dataset with mixed numerical and categorical features.

    The dataset simulates a manufacturing scenario with different product families,
    each characterized by geometric dimensions (numerical) and process-related
    attributes (categorical). The natural cluster structure makes it suitable
    for testing mixed-type clustering algorithms.

    Parameters:
    n_samples (int): Total number of samples to generate. Default is 300.
    random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
    tuple:
        - data (pd.DataFrame): Generated dataset with part_id as index.
        - num_columns (list of str): Names of numerical columns.
        - cat_columns (list of str): Names of categorical columns.
    """
    rng = np.random.default_rng(random_state)

    ### define product families with characteristic distributions
    families = [
        {
            'name': 'small_cylindrical',
            'n': n_samples // 5,
            'length': (15, 5), 'width': (12, 3), 'height': (12, 3),
            'diameter': (20, 4), 'weight': (0.3, 0.1),
            'material': ['steel', 'steel', 'aluminum', 'steel', 'brass'],
            'process': ['turning', 'turning', 'turning', 'grinding', 'turning'],
            'shape': ['cylindrical', 'cylindrical', 'cylindrical', 'cylindrical', 'conical'],
            'surface': ['polished', 'polished', 'raw', 'polished', 'coated'],
        },
        {
            'name': 'large_prismatic',
            'n': n_samples // 5,
            'length': (120, 20), 'width': (80, 15), 'height': (60, 12),
            'diameter': (5, 3), 'weight': (8.5, 2.0),
            'material': ['steel', 'steel', 'steel', 'aluminum', 'steel'],
            'process': ['milling', 'milling', 'drilling', 'milling', 'milling'],
            'shape': ['prismatic', 'prismatic', 'prismatic', 'prismatic', 'complex'],
            'surface': ['raw', 'coated', 'raw', 'raw', 'coated'],
        },
        {
            'name': 'precision_parts',
            'n': n_samples // 5,
            'length': (45, 8), 'width': (30, 6), 'height': (25, 5),
            'diameter': (35, 6), 'weight': (1.2, 0.3),
            'material': ['brass', 'aluminum', 'brass', 'brass', 'aluminum'],
            'process': ['turning', 'grinding', 'turning', 'grinding', 'turning'],
            'shape': ['conical', 'cylindrical', 'conical', 'conical', 'cylindrical'],
            'surface': ['polished', 'polished', 'polished', 'coated', 'polished'],
        },
        {
            'name': 'heavy_components',
            'n': n_samples // 5,
            'length': (200, 30), 'width': (150, 25), 'height': (100, 20),
            'diameter': (80, 15), 'weight': (25.0, 5.0),
            'material': ['steel', 'steel', 'steel', 'steel', 'cast_iron'],
            'process': ['milling', 'drilling', 'milling', 'turning', 'milling'],
            'shape': ['complex', 'prismatic', 'complex', 'complex', 'prismatic'],
            'surface': ['raw', 'raw', 'coated', 'raw', 'raw'],
        },
        {
            'name': 'flat_plates',
            'n': n_samples - 4 * (n_samples // 5),
            'length': (100, 15), 'width': (100, 15), 'height': (8, 2),
            'diameter': (3, 2), 'weight': (2.0, 0.5),
            'material': ['aluminum', 'aluminum', 'steel', 'aluminum', 'plastic'],
            'process': ['milling', 'cutting', 'cutting', 'milling', 'cutting'],
            'shape': ['prismatic', 'prismatic', 'prismatic', 'flat', 'flat'],
            'surface': ['coated', 'raw', 'coated', 'coated', 'raw'],
        },
    ]

    ### generate samples for each product family
    records = []
    for family in families:
        n = family['n']
        for _ in range(n):
            record = {
                'length': max(1, rng.normal(family['length'][0], family['length'][1])),
                'width': max(1, rng.normal(family['width'][0], family['width'][1])),
                'height': max(1, rng.normal(family['height'][0], family['height'][1])),
                'diameter': max(0, rng.normal(family['diameter'][0], family['diameter'][1])),
                'weight': max(0.01, rng.normal(family['weight'][0], family['weight'][1])),
                'material': rng.choice(family['material']),
                'process': rng.choice(family['process']),
                'shape': rng.choice(family['shape']),
                'surface_treatment': rng.choice(family['surface']),
            }
            # approximate volume from dimensions
            record['volume'] = (
                record['length'] * record['width'] * record['height']
                * rng.uniform(0.3, 0.8)
            )
            records.append(record)

    data = pd.DataFrame(records)

    ### set part ID as index
    data.index = [f'PART-{str(i + 1).zfill(4)}' for i in range(len(data))]
    data.index.name = 'part_id'

    ### round numerical values
    num_columns = ['length', 'width', 'height', 'diameter', 'weight', 'volume']
    data[num_columns] = data[num_columns].round(2)

    cat_columns = ['material', 'process', 'shape', 'surface_treatment']

    return data, num_columns, cat_columns
