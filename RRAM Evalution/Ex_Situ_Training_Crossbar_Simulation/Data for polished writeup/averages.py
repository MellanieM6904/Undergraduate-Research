import pandas as pd

gradient = pd.read_csv("Gradient Based Results.csv")
cga = pd.read_csv("CGA Results.csv")
baldwin = pd.read_csv("Baldwinian Results.csv")
lamarck = pd.read_csv("Lamarckian Results.csv")
a_baldwinian = pd.read_csv("Adaptive Baldwinian Results.csv")
a_lamarck = pd.read_csv("Adaptive Lamarckian Results.csv")

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

f.write("ADAPTIVE BALDWINIAN:\n")
f.write(f"Average success deviation: {a_baldwinian['Success Deviation'].mean()}\n")
f.write(f"Average success accuracy: {a_baldwinian['Success Accuracy'].mean()}\n")
f.write(f"Average failure deviation: {a_baldwinian['Failure Deviation'].mean()}\n")
f.write(f"Average failure accuracy: {a_baldwinian['Failure Accuracy'].mean()}\n")

f.write("ADAPTIVE LAMARCKIAN:\n")
f.write(f"Average success deviation: {a_lamarck['Success Deviation'].mean()}\n")
f.write(f"Average success accuracy: {a_lamarck['Success Accuracy'].mean()}\n")
f.write(f"Average failure deviation: {a_lamarck['Failure Deviation'].mean()}\n")
f.write(f"Average failure accuracy: {a_lamarck['Failure Accuracy'].mean()}\n")

f.close()

print("Done.")