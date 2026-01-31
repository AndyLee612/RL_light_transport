import numpy as np
from tqdm import tqdm 

def load_light_transport_npz(file_path):
    """
    Load rl_reward_batch_*.npz and convert into (s, a, r, s_next)
    with:
        state      s      = (position[3], normal[3])         = 6D
        action     a      = direction (wo_world)             = 3D
        reward     r      = radiance (RGB)                   = 3D
        next state s_next = next (pos, normal) OR zero (6D)
    Terminal transitions have:
        s_next = zeros(6)
    """

    data = np.load(file_path)

    positions  = data["positions"]     # (N, 3)
    normals    = data["normals"]       # (N, 3)
    directions = data["directions"]    # (N, 3)
    radiance   = data["radiance"]      # (N, 3)
    ray_ids    = data["ray_ids"]       # (N,)
    bounce_ids = data["bounce_ids"]    # (N,)

    # ----------- Build 6D state -----------
    s = np.concatenate([positions, normals], axis=1)   # (N, 6)

    # ----------- Action (3D) -----------
    a = directions.copy()                              # (N, 3)

    # ----------- Reward (3D) -----------
    r = radiance.copy()                                # (N, 3)

    # Build structured transitions for sorting
    N = s.shape[0]
    transitions = np.zeros(N, dtype=[
        ('ray', np.int64),
        ('bounce', np.int64),
        ('s', float, (6,)),
        ('a', float, (3,)),
        ('r', float, (3,))
    ])

    transitions['ray']    = ray_ids
    transitions['bounce'] = bounce_ids
    transitions['s']      = s
    transitions['a']      = a
    transitions['r']      = r

    # Sort by ray, then bounce
    transitions.sort(order=['ray', 'bounce'])

    # ----------- Build transitions: (s, a, r, s_next) -----------
    s_list = []
    a_list = []
    r_list = []
    s_next_list = []

    for i in tqdm(range(N), desc="proc"):
        s_t = transitions['s'][i]
        a_t = transitions['a'][i]
        r_t = transitions['r'][i]
        ray_t = transitions['ray'][i]
        bounce_t = transitions['bounce'][i]

        # Check if next record is same ray & next bounce
        if i < N - 1:
            ray_next = transitions['ray'][i + 1]
            bounce_next = transitions['bounce'][i + 1]

            if ray_next == ray_t and bounce_next == bounce_t + 1:
                # Valid continuation
                s_next = transitions['s'][i + 1]
            else:
                # Terminal: ray died after this bounce
                s_next = np.zeros(6, dtype=float)
        else:
            # Last record is always terminal
            s_next = np.zeros(6, dtype=float)

        s_list.append(s_t)
        a_list.append(a_t)
        r_list.append(r_t)
        s_next_list.append(s_next)

    # Convert to numpy arrays
    S      = np.array(s_list)       # (N, 6)
    A      = np.array(a_list)       # (N, 3)
    R      = np.array(r_list)       # (N, 3)
    S_next = np.array(s_next_list)  # (N, 6)

    return S, A, R, S_next

if __name__ == "__main__":
    file_path = "rl_reward_batch_0.npz"
    s, a, r, s_next = load_light_transport_npz(file_path)

    print("states:", s.shape)
    print("actions:", a.shape)
    print("rewards:", r.shape)
    print("next_states:", s_next.shape)

    print("first state:", s[0])
    print("first action:", a[0])
    print("first reward:", r[0])
    print("first next_state:", s_next[0])
