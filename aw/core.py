import numpy as np

def direction(neutral,target,k=2,epsilon=1e-8):
    x=np.asarray(neutral,dtype=np.float64); y=np.asarray(target,dtype=np.float64)
    if x.shape!=y.shape or x.ndim!=2:raise ValueError('Paired images must match')
    raw=(y-x).mean(0)
    _,_,vt=np.linalg.svd(x-x.mean(0),full_matrices=False)
    q=vt[:k];v=raw-q.T@(q@raw)
    norm=np.linalg.norm(v)
    if not np.isfinite(norm) or norm<=epsilon:raise ValueError('Degenerate direction')
    return (v/norm).astype('float32'),{'raw_norm':float(np.linalg.norm(raw)),'projected_norm':float(norm),'removed_pcs':k}

def degeneration(text):
    words=text.split()
    if not words:return True
    grams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    return bool(grams and 1-len(set(grams))/len(grams)>.5)

def choose_window(gain_by_layer, width):
    gains = np.asarray(gain_by_layer, dtype=float)
    if gains.ndim != 1 or not np.isfinite(gains).all() or not 1 <= width <= len(gains):
        raise ValueError("Need finite layer responses and valid window width")
    scores = np.array([gains[s:s+width].mean() for s in range(len(gains)-width+1)])
    start = int(np.argmax(scores))
    return {"status": "VALIDATION_SELECTED", "layers": list(range(start+1, start+width+1)), "selection_score": float(scores[start])}

