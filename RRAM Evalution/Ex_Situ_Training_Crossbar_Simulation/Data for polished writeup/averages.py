import pandas as pd

gradient = pd.read_csv("Gradient Based results.csv")
cga = pd.read_csv("CGA Results.csv")
baldwin = pd.read_csv("Baldwinian Results.csv")
lamarck = pd.read_csv("Lamarckian Results.csv")

f = open("Averages.txt", "w")

f.write("GRADIENT:\n")
f.write(f"Average success deviation: {gradient['Success Deviation'].mean()}\n")
f.write(f"Average success accuracy: {gradient['Success Accuracy'].mean()}\n")
f.write(f"Average failure deviation: {gradient['Failure Deviation'].mean()}\n")
f.write(f"Average failure accuracy: {gradient['Failure Accuracy'].mean()}\n")

f.write("CGA:\n")
f.write(f"Average success deviation: {cga['Success Deviation'].mean()}\n")
f.write(f"Average success accuracy: {cga['Success Accuracy'].mean()}\n")
f.write(f"Average failure deviation: {cga['Failure Deviation'].mean()}\n")
f.write(f"Average failure accuracy: {cga['Failure Accuracy'].mean()}\n")

f.write("BALDWINIAN:\n")
f.write(f"Average success deviation: {baldwin['Success Deviation'].mean()}\n")
f.write(f"Average success accuracy: {baldwin['Success Accuracy'].mean()}\n")
f.write(f"Average failure deviation: {baldwin['Failure Deviation'].mean()}\n")
f.write(f"Average failure accuracy: {baldwin['Failure Accuracy'].mean()}\n")

f.write("LAMARCKIAN:\n")
f.write(f"Average success deviation: {lamarck['Success Deviation'].mean()}\n")
f.write(f"Average success accuracy: {lamarck['Success Accuracy'].mean()}\n")
f.write(f"Average failure deviation: {lamarck['Failure Deviation'].mean()}\n")
f.write(f"Average failure accuracy: {lamarck['Failure Accuracy'].mean()}\n")

f.close()

print("Done.")