piece2id = {
    'cube': 1,
    'cylinder': 2,
    'none': 0,
}

id2piece = {
    v: k for k, v in piece2id.items()
}

piece2player = {
    'cube': 'human',
    'cylinder': 'robot',
    'none': 'nobody',
}
