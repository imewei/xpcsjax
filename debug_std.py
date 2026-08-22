import jax
import jax.numpy as jnp

def f(x):
    return jnp.std(x)

x = jnp.array([0.5, 0.5, 0.5, 0.5])
print("f(x)=", f(x))
print("grad=", jax.grad(f)(x))
print("hessian=", jax.hessian(f)(x))

def g(x):
    # mimic compute_regularization_jax cv branch
    mean_val = jnp.mean(x)
    std_val = jnp.std(x)
    mean_is_nonzero = jnp.abs(mean_val) > 1e-10
    safe_denom = jnp.where(mean_is_nonzero, jnp.abs(mean_val), 1.0)
    cv = jnp.where(mean_is_nonzero, std_val / safe_denom, std_val)
    return cv**2

print("g(x)=", g(x))
print("grad g=", jax.grad(g)(x))
print("hessian g=", jax.hessian(g)(x))
