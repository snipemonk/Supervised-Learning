# implementing on a sample dataset

from sklearn.datasets import load_iris

X,y = load_iris(return_X_y = True)

# Run the model
model,report,accuracy = train_decision_tree(X,y,test_size=0.5,max_depth=3)

print("Model Accuracy:", accuracy)
print("Classification Report:\n",report)