**render mode in gym.make**-> determines how env is visualised. 
                            *human* mode shows us a sim. 
                            *rgb array* returns current state as a numerical array(pixels). ideal if we wan to feed the animation in a neural network directly. 
                            *none* runs sim in bg and doesnt show us. it is fast and so good if you want to run a lot of episodes.
                            *ansi* gives in string or text based grid in console

**env.action_space.sample()**-> chooses a random action from the defined actions.
**env.step(action)**-> advances env by one timestep by doing chosen action.
                        outputs *observations*: state data or visual frame of env
                                *reward*
                                *terminated*: is bool, if agent reached natural end state or not (like achieved goal or reached failure)
                                *truncated*: is bool, shows if agent was cutoff. could be due to exceeding time limit or something else
                                *info*: is dict, contains arbitrary debugging and diagnostic details