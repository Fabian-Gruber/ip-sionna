import tensorflow as tf

class genie_mmse_estimator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    def __call__(self, y, no, C, pilot):
        
        print('no: ', no)
                                
        # noise_var = no^2 * I. Be careful of the data types!
        noise_var = tf.cast(no * tf.eye(C.shape[-1], batch_shape=[C.shape[0]]), dtype=tf.complex64)
                                        
        # scaled_C = |p|^2 * C
        scaled_C = tf.cast(tf.math.abs(pilot) ** 2, dtype=tf.complex64) * tf.cast(C, dtype=tf.complex64)
        
        # print('check if scaled_C is hermitian', tf.reduce_all(tf.equal(scaled_C, tf.math.conj(tf.transpose(scaled_C, perm=[0, 2, 1])))))
        
        eigenvalues, eigenvectors = tf.linalg.eigh(scaled_C)
                        
        # Compute the inverse of (Lambda * noise_var)^-1
        inverse_lambda_noise_var = tf.linalg.inv(tf.linalg.diag(eigenvalues) + noise_var)

        # Compute the inverse of the sum
        # inverse = tf.matmul(tf.matmul(eigenvectors, inverse_lambda_noise_var), tf.transpose(eigenvectors, conjugate=True, perm=[0, 2, 1]))
        
        # print('shape of einsum of inverse_lambda_noise_var: ', tf.einsum('ijk,ilk->ijl', inverse_lambda_noise_var, tf.transpose(eigenvectors, conjugate=True, perm=[0, 1, 2])).shape)
        
        inverse = tf.einsum('ijk,ikl->ijl', eigenvectors, tf.einsum('ijk,ilk->ijl', inverse_lambda_noise_var, tf.transpose(eigenvectors, conjugate=True, perm=[0, 1, 2])))
        
        # print('where are inverse and inverse_2 the same: ', tf.reduce_sum(tf.where(tf.abs(inverse - inverse_2) > 1e-2, tf.ones_like(inverse), tf.zeros_like(inverse))))
        
        print('eigenvalues: ', eigenvalues[0, :])
        print('maximum value of eigenvalues: ', tf.math.reduce_max(tf.math.real(eigenvalues), axis=1)[0])
        print('minimum value of eigenvalues: ', tf.math.reduce_min(tf.math.real(eigenvalues), axis=1)[0])  
        
        cond_number = tf.math.reduce_max(tf.math.real(eigenvalues), axis=1)[0] / tf.math.reduce_min(tf.math.real(eigenvalues), axis=1)[0]
        print('condition number: ', cond_number)
        
        
        # print('inverted inverse: ', tf.matmul(tf.linalg.pinv(C), C)[0,0,0])
        
        I = tf.eye(C.shape[-1], batch_shape=[C.shape[0]], dtype=tf.complex64)
        
        C_inv_approx = gradient_descent_solve(C, I, iterations=1000, lr=1e-3)
        
        print('C_inv_approx times C: ', tf.matmul(C_inv_approx, C)[0,0,0])
                                                
        # scaled_C_2 = conj(p) * C
        scaled_C_2 = tf.math.conj(pilot) * tf.cast(C, dtype=tf.complex64)
                        
        # matrix = scaled_C_2 * inverse
        matrix = tf.matmul(scaled_C_2, inverse)
        matrix_2 = tf.einsum('ijk,ikl->ijl', scaled_C_2, inverse)
        print('where are matrix and matrix_2 the same: ', tf.reduce_sum(tf.where(tf.abs(matrix - matrix_2) > 1e-2, tf.ones_like(matrix), tf.zeros_like(matrix))))
                                                                
        print('matrix elements: ', matrix[0,:5,:5])
                                                                
        # h_hat_mmse = (scaled_C_2 * inverse) * y. Be careful of the data types!
        h_hat_mmse = tf.matmul(matrix, tf.transpose(tf.cast(y, dtype=tf.complex64), perm=[0, 2, 1]))   
        h_hat_mmse_2 = tf.einsum('ijk,ilk->ijl', matrix, y)     
        
        # h_hat_per_hand = tf.matmul(matrix[0, :, :], tf.squeeze(y[0, 0, :]))
        
        print('where are h_hat_mmse and h_hat_mmse_2 the same', tf.reduce_sum(tf.where(tf.abs(h_hat_mmse - h_hat_mmse_2) > 1e-2, tf.ones_like(h_hat_mmse), tf.zeros_like(h_hat_mmse))))
                                                
        return tf.transpose(h_hat_mmse, perm=[0, 2, 1])
    
def gradient_descent_solve(A, b, iterations=1000, lr=1e-3):
    """
    Solves the linear system Ax = b using gradient descent for batched matrices.
    
    Args:
    - A: Batch of complex-valued matrices with shape [batch_size, n, n]
    - b: Batch of matrices (can be the identity matrix for inverse computation) with shape [batch_size, n, n]
    - iterations: Number of iterations to run the algorithm
    - lr: Learning rate

    Returns:
    - x: Solution to the linear system for each batch
    """
    
    # Initialize x's real and imaginary parts separately
    real_part = tf.random.normal(b.shape, dtype=tf.float32)
    imag_part = tf.random.normal(b.shape, dtype=tf.float32)

    # Combine the real and imaginary parts to get the complex tensor
    x = tf.Variable(tf.complex(real_part, imag_part))
    
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            # The loss is computed batch-wise and then averaged
            loss = tf.reduce_mean(tf.abs(tf.matmul(A, x) - b))
        
        grads = tape.gradient(loss, x)
        x.assign_sub(lr * grads)
    
    return x