## State Representations as Incentives for Reinforcement Learning Agents: A Sim2Real Analysis on Robotic Grasping
---

<p align="center">
  <img src="final_video.gif" width="600" height="350"/>
</p>

<p align="center">
  <img src="cover-picture.png" width="600" height="250"/>
</p>

The VTPRL simulator and RL agents base implementations can be found at: https://github.com/tum-i6/VTPRL


----
Table I: Mean success rate across the different state representation strategies
| Strategy   | Average + Std. (Idl. Sim.) | Best Model (Idl. Sim.) | Best Model (Rnd. Sim.) | Best Model (Real) |
|------------|-----------------------------|------------------------|------------------------|-------------------|
| Ruckig     | **100%**                    | **100%**               | N/A                    | **100%**          |
| St.        | **100%**                    | **100%**               | N/A                    | **100%**          |
| St. (rnd.) | **100%**                    | **100%**               | N/A                    | **100%**          |
| VtS        | 91.6% ± 2.2                 | 94%                    | 70%                    | 52%               |
| IGAE       | 96.0% ± 2.8                 | **100%**               | 78%                    | **84%**          |
| AE         | 82.4% ± 7.1                 | 92%                    | 70%                    | 60%               |
| EtE        | 54.8% ± 11.0                | 78%                    | 44%                    | 24%               |


Table II: Evaluation of autoencoder-based vision models over KNN-MSE criterion
| Strategy | Mean   | Std.   | Max   | Min                     |
|----------|--------|--------|-------|-------------------------|
| IGAE     | **0.0393** | **0.1679** | 1.8883 | **1.4122 x 10^-6** |
| AE       | 0.0459 | 0.1839 | 1.6960 | **1.4122 x 10^-6** |
| VtS      | 0.0488 | 0.1946 | **1.6857** | 1.8881 x 10^-6 |


### Citing the Project
---

To cite this repository in publications:
```bibtex
@misc{2023representation,
      title={Representation Abstractions as Incentives for Reinforcement Learning Agents: A Robotic Grasping Case Study}, 
      author={Panagiotis Petropoulakis and Ludwig Gräf and Josip Josifovski and Mohammadhossein Malmir and Alois Knoll},
      year={2023},
      eprint={2309.11984},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
### Acknowledgments
---
* Ludwig Gräf - vision-based implementations

