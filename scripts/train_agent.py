import agents.agent as agent
# This script initializes and trains a random forest with specified parameters.
agent = agent.HangmanAgent(
    total_guesses_allowed=6,
    sample_data_size=50000,
    n_estimators=200,
    max_depth=30
)
# Generate training data for the agent
print("Generating training data...")
agent.data()
# Train the agent's model using the generated data
print("Training the model...")
agent.train_model()
# Plot the learning curve to visualize the training process
print("Plotting learning curve...")
agent.plot_learning_curve()