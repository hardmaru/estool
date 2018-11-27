# ESTool

<center>
<img src="https://cdn.rawgit.com/hardmaru/pybullet_animations/f6f7fcd7/anim/biped/biped_cma.gif" width="100%"/>
<i>Evolved Biped Walker.</i><br/>
</center>
<p></p>

Implementation of various Evolution Strategies, such as GA, PEPG, CMA-ES and OpenAI's ES using common interface.

CMA-ES is wrapping around [pycma](https://github.com/CMA-ES/pycma).

# Notes

The tool last tested using the following configuration:

- TensorFlow 1.8.0

- NumPy 1.13.3 (1.14 has some annoying warning).

- OpenAI Gym 0.9.4 (breaks for 1.0+ since they changed the API).

- cma 2.2.0, basically 2+ should work.

- PyBullet 1.6.3 (possible that newer versions might work, but have not tested).

- Python 3, although 2 might work.

- mpi4py 2

## Backround Reading:

[A Visual Guide to Evolution Strategies](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/)

[Evolving Stable Strategies](http://blog.otoro.net/2017/11/12/evolving-stable-strategies/)

## Using Evolution Strategies Library

To use es.py, please check out the `simple_es_example.ipynb` notebook.

The basic concept is:

```
solver = EvolutionStrategy()
while True:

  # ask the ES to give us a set of candidate solutions
  solutions = solver.ask()

  # create an array to hold the solutions.
  # solver.popsize = population size
  rewards = np.zeros(solver.popsize)

  # calculate the reward for each given solution
  # using your own evaluate() method
  for i in range(solver.popsize):
    rewards[i] = evaluate(solutions[i])

  # give rewards back to ES
  solver.tell(rewards)

  # get best parameter, reward from ES
  reward_vector = solver.result()

  if reward_vector[1] > MY_REQUIRED_REWARD:
    break
```

## Parallel Processing Training with MPI

Please read [Evolving Stable Strategies](http://blog.otoro.net/2017/11/12/evolving-stable-strategies/) article for more demos and use cases.

To use the training tool (relies on MPI):

```
python train.py bullet_racecar -n 8 -t 4
```

will launch training jobs with 32 workers (using 8 MPI processes). the best model will be saved as a .json file in log/. This model should train in a few minutes on a 2014 MacBook Pro.

If you have more compute and have access to a 64-core CPU machine, I recommend:

```
python train.py name_of_environment -e 16 -n 64 -t 4
```

This will calculate fitness values based on an average of 16 random runs, on 256 workers (64 MPI processes x 4). In my experience this works reasonably well for most tasks inside `config.py`.

After training, to run pre-trained models:

```
python model.py bullet_ant log/name_of_your_json_file.json
```

<center>
<!--<img src="{{ site.baseurl }}/assets/20171109/biped/bipedcover.gif" width="100%"/><br/>-->
<!--<img src="{{ site.baseurl }}/assets/20171109/kuka/kuka.gif" width="100%"/><br/>-->
<img src="https://cdn.rawgit.com/hardmaru/pybullet_animations/f6f7fcd7/anim/robo/bullet_ant_demo.gif" width="50%"/><br/>
<i>bullet_ant pybullet environment. PEPG.</i><br/>
</center>
<p></p>

Another example: to run a minitaur duck model, run this locally:

```
python model.py bullet_minitaur_duck zoo/bullet_minitaur_duck.cma.256.json
```

<center>
<!--<img src="{{ site.baseurl }}/assets/20171109/biped/bipedcover.gif" width="100%"/><br/>-->
<!--<img src="{{ site.baseurl }}/assets/20171109/kuka/kuka.gif" width="100%"/><br/>-->
<img src="https://cdn.rawgit.com/hardmaru/pybullet_animations/8a6ccaf5/anim/minitaur/duck_normal_small.gif" width="100%"/><br/>
<i>Custom Minitaur Env.</i><br/>
</center>
<p></p>


In the .hist.json file, and on the screen output, we track the progress of training. The ordering of fields are:

- generation count
- time (seconds) taken so far
- average fitness
- worst fitness
- best fitness
- average standard deviation of params
- average timesteps taken
- max timesteps taken

Using `plot_training_progress.ipynb` in an IPython notebook, you can plot the traning logs for the `.hist.json` files. For example, in the `bullet_ant` task:

<center>
<img src="https://cdn.rawgit.com/hardmaru/pybullet_animations/5a3847d0/svg/bullet_ant.svg" width="100%"/><br/>
<i>Bullet Ant training progress.</i><br/>
</center>
<p></p>

You need to install mpi4py, pybullet, gym etc to use various environments. Also roboschool/Box2D for some of the OpenAI gym envs.

On Windows, it is easiest to install mpi4py as follows:

- Download and install mpi_x64.Msi from the HPC Pack 2012 MS-MPI Redistributable Package
- Install a recent Visual Studio version with C++ compiler
- Open a command prompt
```
git clone https://github.com/mpi4py/mpi4py
cd mpi4py
python setup.py install
```
Modify the train.py script and replace mpirun with mpiexec and -np with -n


