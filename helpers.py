def interpol_values_S(paths, t, dt):
    if t == .0:
        return paths[:,0]
    elif t == len(paths[0]) * dt:
        return paths[:,-1]
    elif t < .0 or t > len(paths[0]) * dt:
        raise Exception('Wrong value!')
    else:
        for i in range(1, len(paths[0]) - 1):
            if (i - 1) * dt < t and i * dt >= t:
                s = (t - (i - 1) * dt) / dt
                return (1.0 - s) * paths[:,i-1] + s * paths[:,i]

def interpol_values_B(values, t, dt):
    if t == .0:
        return values[0]
    elif t == len(values) * dt:
        return values[-1]
    elif t < .0 or t > len(values) * dt:
        raise Exception('Wrong value!')
    else:
        for i in range(1, len(values) - 1):
            if (i - 1) * dt < t and i * dt >= t:
                s = (t - (i - 1) * dt) / dt
                return (1.0 - s) * values[i-1] + s * values[i]

