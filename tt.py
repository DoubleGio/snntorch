import torch, snntorch as snn

def spike_steps(reset, T=32, **kw):
    n = snn.Leaky(beta=0.8, reset_mechanism=reset, **kw)
    mem, steps = n.init_leaky(), []
    for t in range(T):
        spk, mem = n(torch.full((1,), 0.3), mem)
        if spk.item(): steps.append(t)
    return steps

print("zero,  reset_delay=True :", spike_steps("zero"))
print("zero,  reset_delay=False:", spike_steps("zero", reset_delay=False))
print("subtract               :", spike_steps("subtract"))
try: snn.Leaky(beta=0.8, reset_mechanism="bogus")
except ValueError as e: print("bad-name error:", e)
try:
    n = snn.Synaptic(alpha=0.9, beta=0.8, reset_mechanism="subtract_beta")
    n(torch.full((1, 1), 0.9), *n.init_synaptic()); print("Synaptic subtract_beta: ok")
except Exception as e: print("Synaptic subtract_beta:", type(e).__name__, e)