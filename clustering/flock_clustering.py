# Inspired by
# https://agentpy.readthedocs.io/en/latest/agentpy_flocking.html
#

import agentpy as ap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize(v):
    """ Normalize a vector to length 1. """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class Boid(ap.Agent):
    """ An agent with a position and velocity in a continuous space,
    who follows Craig Reynolds three rules of flocking behavior;
    plus a fourth rule to avoid the edges of the simulation space. """

    def setup(self):

        self.velocity = normalize(
            self.model.nprandom.random(self.p.ndim) - 0.5)
        self.records = self.p.athlete_groups[self.id - 2]

    def setup_pos(self, space):

        self.space = space
        self.pos = space.positions[self]

    def update_velocity(self):

        pos = self.pos
        ndim = self.p.ndim

        # Rule 1 - Cohesion
        nbs = self.get_neighbors(self.p.outer_radius)
        nbs_len = len(nbs)
        nbs_pos_array = np.array([other.pos for other in nbs])
        nbs_vec_array = np.array([other.velocity for other in nbs])
        if nbs_len > 0:
            center = np.sum(nbs_pos_array, 0) / nbs_len
            v1 = (center - pos) * self.p.cohesion_strength
        else:
            v1 = np.zeros(ndim)

        # Rule 2 - Seperation
        v2 = np.zeros(ndim)
        for nb in self.get_neighbors(self.p.inner_radius):
            v2 -= nb.pos - pos
        v2 *= self.p.seperation_strength

        # Rule 3 - Alignment
        if nbs_len > 0:
            average_v = np.sum(nbs_vec_array, 0) / nbs_len
            v3 = (average_v - self.velocity) * self.p.alignment_strength
        else:
            v3 = np.zeros(ndim)

        # Rule 4 - Borders
        v4 = np.zeros(ndim)
        d = self.p.border_distance
        s = self.p.border_strength
        for i in range(ndim):
            if pos[i] < d:
                v4[i] += s
            elif pos[i] > self.space.shape[i] - d:
                v4[i] -= s

        # Update velocity
        self.velocity += v1 + v2 + v3 + v4
        self.velocity = normalize(self.velocity)

    def update_position(self):

        self.space.move_by(self, self.velocity)

    def get_distance(self, other):
        time = self.model.t
        # get current record where date column is equal to simulation time
        self_current_record = self.records[self.records['Date'] == time - 1]
        other_current_record = other.records[other.records['Date'] == time - 1]

        if (self_current_record.empty or other_current_record.empty):
            return np.inf

        # drop date and athlete id columns
        self_current_record = self_current_record.drop(
            columns=['Date', 'Athlete ID'])
        other_current_record = other_current_record.drop(
            columns=['Date', 'Athlete ID'])

        diff = np.array(self_current_record) - np.array(other_current_record)
        diff_norm = np.linalg.norm(diff)
        return diff_norm

    def get_neighbors(self, distance):
        return [other for other in self.space.agents if self.get_distance(other) < distance]


class BoidsModel(ap.Model):
    """
    An agent-based model of animals' flocking behavior,
    based on Craig Reynolds' Boids Model [1]
    and Conrad Parkers' Boids Pseudocode [2].

    [1] http://www.red3d.com/cwr/boids/
    [2] http://www.vergenet.net/~conrad/boids/pseudocode.html
    """

    def setup(self):
        """ Initializes the agents and network of the model. """

        self.space = ap.Space(self, shape=[self.p.size]*self.p.ndim)
        self.agents = ap.AgentList(self, self.p.population, Boid)
        self.space.add_agents(self.agents, random=True)
        self.agents.setup_pos(self.space)

    def step(self):
        """ Defines the models' events per simulation step. """

        self.agents.update_velocity()  # Adjust direction
        self.agents.update_position()  # Move into new direction

        if (self.model.t == self.p.steps):
            # save positions to file
            positions = [(boid.records['Athlete ID'].iloc[0], position[0], position[1]) for boid, position in self.space.positions.items()]
            
            df = pd.DataFrame(positions, columns=['Athlete ID', 'x', 'y'])
            df.to_csv('clustering/boids_positions.csv', index=False)


def animation_plot_single(m, ax):
    ndim = m.p.ndim
    ax.set_title(f"Boids Flocking Model {ndim}D t={m.t}")
    pos = m.space.positions.values()
    pos = np.array(list(pos)).T  # Transform
    ax.scatter(*pos, s=1, c='black')
    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    if ndim == 3:
        ax.set_zlim(0, m.p.size)
    ax.set_axis_off()


def animation_plot(m, p):
    projection = '3d' if p['ndim'] == 3 else None
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection=projection)
    animation = ap.animate(m(p), fig, ax, animation_plot_single)
    animation.save("boids_2.gif")
    return animation


if __name__ == "__main__":
    # Load daily data
    data = pd.read_csv('data/day_approach_maskedID_timeseries.csv')

    # Filter columns from previous days
    keep_cols = data.columns[data.columns.str.endswith('.6') |
                             data.columns.isin(['Athlete ID', 'injury', 'Date'])]
    data = data.loc[:, keep_cols]

    # Group data by Athlete ID without aggregation
    athlete_groups = {athlete_id: group for athlete_id,
                      group in data.groupby('Athlete ID')}

    parameters2D = {
        'size': 50,
        'seed': 123,
        'steps': 200,
        'ndim': 2,
        'population': len(athlete_groups),
        'inner_radius': 3,
        'outer_radius': 10,
        'border_distance': 10,
        'cohesion_strength': 0.005,
        # 'seperation_strength': 0.1,
        'seperation_strength': 0.0001,
        'alignment_strength': 0.3,
        'border_strength': 0.5,
        'athlete_groups': athlete_groups,
    }

    animation_plot(BoidsModel, parameters2D)
