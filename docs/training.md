# Training

## Teacher Network Training

To reproduce the paper's result

```
cd train
bash script/train_teacher.sh <model_name> <dataset_name>
```

## Student Network Training

To reproduce the paper's result

### VXP

Make sure to configure the correct path for the teacher network.

```
cd train
bash script/train_student_firststage.sh <dataset_name>
```

Once the training is done, please configure the correct file path the in setup file, and then perform second-stage training.

```
bash script/train_student_secondstage.sh <dataset_name>
```

### Cattaneo's

Make sure to configure the correct path for the teacher network.

```
cd train
bash script/train_student.sh <dataset_name>
```
