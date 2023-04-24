import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)

def driving_force(position, velocity, particles, num):
    # This is an example driving force with attraction between particles
    F_d = np.array([0.0, 0.0, 0.01])
    F_a = np.array([0, 0, 0], dtype=np.float64)
    G = 1  # gravitational constant
    jj = particles[num]
    for i in range(len(particles)):
        # print('---',i,num)
        if i != num:
            ii = particles[i]
            # print(ii)
            r = np.linalg.norm(position - ii.position)
            # print('r',r)
            F_a += G*jj.mass*ii.mass*(ii.position - position)/(r**3)
    print('fa',F_a)


    return F_d + F_a

# Initialize the simulation parameters
num_particles = 6
dt = 0.01

# Initialize the particles
particles = []
for i in range(num_particles):
    if i==1:
        mass = 10000
    mass = 1
    position = [i+1, 0, 0]
    velocity = [0, i+1, 0]
    particle = Particle(mass, position, velocity)
    particles.append(particle)

# Initialize the track arrays for each particle
x = [np.array([particle.position[0]], dtype=np.float64) for particle in particles]
y = [np.array([particle.position[1]], dtype=np.float64) for particle in particles]
z = [np.array([particle.position[2]], dtype=np.float64) for particle in particles]

# Initialize the velocity arrays for each particle
vx = [np.array([particle.velocity[0]], dtype=np.float64) for particle in particles]
vy = [np.array([particle.velocity[1]], dtype=np.float64) for particle in particles]
vz = [np.array([particle.velocity[2]], dtype=np.float64) for particle in particles]

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Run the simulation
for t in np.arange(0, 4, dt):
    for i, particle in enumerate(particles):
        # print(i)
        # Calculate the acceleration of the particle
        F = driving_force(particle.position, particle.velocity, particles, i)
        a = F/particle.mass

        # Update the velocity and position of the particle
        particle.velocity += a*dt
        particle.position += particle.velocity*dt

        # Add the current position and velocity to the track arrays
        x[i] = np.concatenate((x[i], np.array([particle.position[0]], dtype=np.float64)), axis=0)
        y[i] = np.concatenate((y[i], np.array([particle.position[1]], dtype=np.float64)), axis=0)
        z[i] = np.concatenate((z[i], np.array([particle.position[2]], dtype=np.float64)), axis=0)

        vx[i] = np.concatenate((vx[i], np.array([particle.velocity[0]], dtype=np.float64)), axis=0)
        vy[i] = np.concatenate((vy[i], np.array([particle.velocity[1]], dtype=np.float64)), axis=0)
        vz[i] = np.concatenate((vz[i], np.array([particle.velocity[2]], dtype=np.float64)), axis=0)

def update(x,y,z,t,dt):
    tt = int(t/dt)
    ax.plot(x[:tt], y[:tt], z[:tt], color='g')
    plt.show()

# for t in np.arange(0, 1, dt):
#     print(t)
#     update(x,y,z,t,dt)

# Create the animation
# ani = FuncAnimation(fig, update(x,y,z,t,dt), frames=int(1/dt), interval=10)

for i, particle in enumerate(particles):
    ax.plot(x[i], y[i], z[i], label=f"Particle {i+1}")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

