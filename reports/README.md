# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [X] Create a git repository (M5)
* [X] Make sure that all team members have write access to the GitHub repository (M5)
* [X] Create a dedicated environment for you project to keep track of your packages (M2)
* [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [X] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [X] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [X] Do a bit of code typing and remember to document essential parts of your code (M7)
* [X] Setup version control for your data or part of your data (M8)
* [X] Add command line interfaces and project commands to your code where it makes sense (M9)
* [X] Construct one or multiple docker files for your code (M10)
* [X] Build the docker files locally and make sure they work as intended (M10)
* [X] Write one or multiple configurations files for your experiments (M11)
* [ ] Use Hydra to load the configurations and manage your hyperparameters (M11)
* [X] Use profiling to optimize your code (M12)
* [X] Use logging to log important events in your code (M14)
* [X] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [X] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [X] Write unit tests related to the data part of your code (M16)
* [X] Write unit tests related to model construction and or model training (M16)
* [X] Calculate the code coverage (M16)
* [X] Get some continuous integration running on the GitHub repository (M17)
* [X] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [X] Add a linting step to your continuous integration (M17)
* [X] Add pre-commit hooks to your version control setup (M18)
* [X] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [X] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [X] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [X] Create a FastAPI application that can do inference using your model (M22)
* [X] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [X] Write API tests for your application and setup continues integration for these (M24)
* [X] Load test your application (M24)
* [X] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [X] Create a frontend for your API (M26)

### Week 3

* [X] Check how robust your model is towards data drifting (M27)
* [X] Deploy to the cloud a drift detection API (M27)
* [X] Instrument your API with a couple of system metrics (M28)
* [X] Setup cloud monitoring of your instrumented application (M28)
* [X] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [X] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [X] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [X] Make sure all group members have an understanding about all parts of the project
* [X] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 38

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s201680, s214585, s214611, s214647

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used the Huggingface page [PyTorch Image Models (TIMM)](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file) containing different pre-trained Pytorch image encoders and backbones. From this large list we chose to work with a ResNet structure called [ResNet50-D](https://huggingface.co/timm/resnet50d.ra4_e3600_r224_in1k). This model is employed in our project to predict image classes. Here we utilize that the model is already trained on the ImageNet-1k, and thus only a further fine-tuning of the last classification layer is necessary to achieve reliable results. It helped completing the project as we did not have to worry about training a model from scratch, but instead could rely on an easily integratable model.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We initially chose to work with the UV dependency manager as it had shown to be super fast and manage dependencies great across devices. The dependencies are listed in our *pyproject.toml* file, where a new dependency can be added by running the command *uv add ...*, and sorted in dependency groups depending on which case we need to use it. This can for an instance help us reduce the size of our docker images. For a new user to get a complete copy of our environment they first need to install UV on their device. Afterwards, they can run *uv sync which creates a virtual environment (alternatively by by running *uv venv*) and then download all relevant dependencies regardless of operating system. Afterwards *uv run is used to run files in the project. However in the project phase we encountered a few issues, so to easily reuse some of our code from the exercises made with pip we instead chose to list the dependencies in different requirements files. Each corresponds to the specific needs of the docker image it is used to create. To use requirement files on mac first use:
```
 python3 -m venv .venv
 source .venv/bin/activate
 pip install -r requirements_backend.txt
 ```
where you change the name to the relevant requirements file.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

Using the cookiecutter template an overall structure for our repository was achieved dividing the code, saved models, data, tests, and otherfile into different relevant folders. Along the project all relevant data collection, model definition, training, evaluation, and backend are contained in the *src* folder. Following the training, model configurations and exports have been saved in the *models* folder in relevant subfolders for e.g. the sweep for hyperparameter optimization or profiling - though some of these are added to the gitignore due to file sizes being big and not relevant for pushing. The relevant .yaml files for configuring this are found in the *configs* folder along with all cloudbuild files. As the project uses dvc to handle the data, a *.dvc* has been added, while the *data* folder has been added to the .gitignore. In the *.github* folder can all workflows for Github Actions be found, and dockerfiles are also found in their respective folder as laid out by the cookiecutter template. In the root of the repository can the ignore files be found together with all requirements files.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We added two different workflows to Github actions to ensure code quality and code formatting. The first is a continuous integration with pre-commit actions, which automatically pushes any changes found to the main branch. Secondly, a workflow with Ruff linting check and formats the code as well.
All type hints and function or class documentation have been written locally with the help of Github Copilot to give a quick template and initial example to check and iterate on.
The reason both formatting and documentations are important in larger projects, but also in general, is to ensure a cohesive structure, so that everyone can effortlessly understand what others have written and worked on.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:
In total, we have implemented tests for the model, dataloading, API functionality and helper functions used in the API. We have furthermore carried out load testing for the (backend) API. However, due to some issues with the secrets configuration, not all these tests are run in the continuous integration framework - but they have at the very least been run and passed locally.


### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:
It appears that we only have 16% code coverage, but we suspect this to be higher in reality as it does not seem to take the model unit tests into consideration - we were unable to figure out why this happens. In case we did have a higher coverage, you cannot necessarily trust it to be error free - for instance, one might have overlooked edge cases which could give errors if specifically tested.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

In the start of the project we didn't use branches or pull-requests as we had a lot of communication and even pair programming during the project which made it less necessary. Additionally, the start of getting a model loaded, setting up training, or processing the data could all be done in parallel without much of a problem.
However, when we approached the end of the project, and we were working on different solutions, we utilized the advantage of branches to not ruin the working solution on the main branch. At this point we also had set up a great CI/CD pipeline including a testing workflow, which allowed us to check that everything was working before merging the pull request into main.
One of the benefits of working with branches is exactly that you can keep main stable and working while developing new features. Furthermore, it makes it easy to revert back to previous versions and in general work together with many people as you can review each other's code.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

Yes, we used DVC to manage our data, linking it to a Cloud Storage bucket. DVC ensured that the same raw and processed data was available to all team members, reducing errors caused by inconsistencies in data formats or versions. For instance, after splitting and processing the data into PyTorch tensors, these could be easily distributed across the team. This versioning made experiments more reliable and streamlined the entire pipeline.

Additionally, DVC enabled automation by maintaining the data in the cloud, ensuring accessibility and consistency. It improved reproducibility by allowing us to trace specific results back to the exact dataset version used, further enhancing reliability. While our dataset was static during this project, making versioning less critical, DVC’s capabilities would become indispensable if new data were added in the future. Overall, DVC simplified data management and enhanced the efficiency and reproducibility of our workflow.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

As mentioned in an earlier question, our continuous integration  setup is designed to ensure code quality and reliability through automated testing and linting. We have organized our CI into multiple workflows, each serving a specific purpose.
- Unit Testing and Coverage: We run unit tests using pytest and measure code coverage using coverage. This ensures that our code is thoroughly tested and helps identify untested parts of the codebase. The workflow is triggered on every push and pull request to the main branch. We test our code on multiple operating systems (Ubuntu, Windows, and macOS) and Python versions 3.12 to ensure compatibility across different environments. We also use caching to speed up the installation of dependencies.
- Linting: We use ruff for linting our code to ensure it adheres to coding standards and best practices. This helps maintain code quality and readability. The linting step is included in our pre-commit hooks and is also run as part of our CI workflow.
- CML workflow: We use DVC to manage our data and model versions, and we then have a workflow which checks the data statistics of the data. A triggering of the workflow can be seen [here](https://github.com/dtu-degens/MLOps_2025/actions/runs/12934553156/job/36075840417).

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

For running experiments we used a config file to run a sweep for hyperparameter optimization through Weights and Biases. This tested different values for our model during training, which was possible to run due to the addition of the *typer* library, parsing the arguments along to the training loop when the file is called.
The file can also be called a single time through *invoke* by writing *invoke train*. This will call a certain configuration set in the tasks.py file.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

As we used a config file, the relevant configuration can be found there. Also all model runs were saved to wandb (weights and biases) together with their artifacts, so that any of the trained models can be found again. To reproduce the sweep experiment one can run a sweep to wandb themselves, and then call an agent to train the models. Furthermore, we are setting a seed as part of our experiments to ensure that the randomness is tracked. This question however is not super relevant in our case as we did not do many experiments.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

Below are found three images from the sweep we did. The [first image](figures/Sweep_charts.png) shows four graphs over the four metrics that we chose to track during training, this being the loss and accuracy on the training set and validation set. Here we see the progress over the training loop which shows that the model seems to quickly overfit to the training data.The validation metrics also stagnate at the same time at around 60%. This is not too bad of an accuracy considering there are 1000 classes, however, the model did achieve impressive results on the imagenet-1k dataset, so it seems that improvements can still be made to generalize the learning more and avoid overfit.
While we added weight decay to the optimizer to add regularization, it does not seem to have had enough of an impact. It did show to be an important parameter as reflected in [the second image](figures/Sweep_importance.png) on the importance of each parameter. The parameter with the most importance when running the sweep with bayesian optimization when minimizing the validation loss is the learning rate. We also see the history of the models and their validation loss. We see that the sweep investigated different combinations at the start, but later on the models all achieved similar values on the metric. This resulted from the Bayesian optimization having found certain parameter values that work well for the minimization and focusing on those. More exploration would maybe have been ideal in our case to further test many different configurations, and also avoid that a combination could be seen as bad because of an unlucky value from the learning rate or weight decay distribution sampling.
Lastly [the third image](figures/Sweep_diagram.png) shows what combination each run had and what validation loss that resulted in.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For our project, we developed several Docker images to streamline the development and deployment processes. We created mainly two Docker images a backend and a frontend, but furthermore also a data drift detection. These are made to ensure consistency across different environments and to facilitate easy scaling and reproducibility.
- Backend Image: We built a Docker image for the backend, which includes the necessary scripts and dependencies to serve the backend API which is used for inference running our trained model.
- Frontend Image: We created a Docker image for the frontend, which uses Streamlit to provide a user interface for interacting with the model.
- Data Drift Detection Image: We built a Docker image for detecting data drift. This image includes the necessary scripts and dependencies to analyze data drift.
Mostly our docker images were part of workflows or cloudbuild. Therefore, the backend docker image would as an example be built when running `gcloud builds submit --config cloudbuild_backend.yaml`. If would then be run by in the deploy phase as follows `run deploy backend-pokedec --image=europe-west3-docker.pkg.dev/$PROJECT_ID/mlops-container-registry/backend_pokedec:latests --region=europe-west3 --platform=managed`. A link to the docker file is [here](europe-west3-docker.pkg.dev/exalted-strata-447112-s0/mlops-container-registry/backend_pokedec@sha256:a5ad3f1bc7e71b8c55bd6ced6a99594ea682dad45d2eaff1d8d375fb4f9f9302).

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

While the preference for how you prefer to debug code varies between people, all group members are comfortable using the VS Code Python Debugger to execute and check specific lines of code, and test different functionalities before running the entire script. This was especially useful when both processing then data, and setting up the model for training and evaluation.
We also did a profiling of the model during a small training loop. This showed that the dataloader with just 1 worker was not taking up any processing time at all, and therefore it wasn’t needed to parallelize the data. It was hard to use the profiling of the layers in the model to much, as the model architecture is set from Huggingface.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used Google cloud storage buckets for storing the training data and for storing the logs when the backend was called with new images,
this was done to detect data drifting.
Furthermore we also used Google Artifact registry to store our docker containers and we used google Run to host the containers on the web, so they were accessible.
Google Logging and monitoring was also used to track the usage of the services, and IAM, service accounts and secrets manager to manage permissions.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We did not use the compute engine and virtual machines, as we trained our model locally. 


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[Our Buckets](figures/our_buckets.png)
[Inside Our Bucket](figures/our_bucekt_inside.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[Our Registry](figures/our_registry.png)
[Inside our Registry](figures/our_registry_inside.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

[Our build history](figures/our_build_history.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We did not train the model in the cloud. While it would have been possible to do so, we instead utilized the access we had to a local GPU (Nvidia RTX 4070 Ti Super) with enough processing power to do the wandb sweep.
If we had to instead do the training in the cloud, with e.g. Vertex AI, a docker image for the training would instead be built and pushed when submitting the train config. This config would further specify a separate config file with the specifications for the virtual machine, before finally sending the job to Vertex AI in the cloud on this VM. It would lastly also be responsible to parsing along any secrets from the Secret Manager as an wandb api key.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We used FastAPI to make an endpoint for our backend which could then be called by the frontend which was made with streamlit. The structure of the API was very simple consisting of one root endpoint to check the API availability and a post endpoint /classify, taking an image (a file) as input and returning the predicted label and the probability distribution of the labels.
The model is loaded in the backend through wandb’s artifacts using that we can attach an alias to the wanted model. Here the chosen alias is *best*, and only the model with this unique alias will be used in our API. This was achieved by both parsing the wandb api key as a secret to the docker image from the Secret Manager, and exposing the same secret as an environment variable to the Cloud Run service.
One thing we could have done to maybe enhance our API was to maybe do some caching of the model to make the response time even faster first response farster, as it takes some time to download the model from wandb when starting the backend up upon the first request.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We managed to deploy our API in the cloud using the google cloud run, we did so by first containerising the API into an application using docker (locally only writing the docker file). Then we would use gcloud build summit specifying a config file cloubuild_backend containing the steps for first building the image, then pushing the image into the artifact registry and then deploying using cloud run. The link to the hosted API is [here](https://backend-pokedec-228711502156.europe-west3.run.app/docs?fbclid=IwZXh0bgNhZW0CMTEAAR3qRjYHBhRdPM5MuNGGrzR3PEkQklV4vnPjVWMdkpv2LPVHfcz2wQM1AyU_aem_Vwkqx1o5GWBtL_Ua4KBIpw#/)

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We performed testing of the backend API. We tested both the root end point as well as the classify endpoint. We additionally tested some edge cases such as providing the API with an unallowed input (i.e. a .txt file even though it only accepts images) as well as no input at all.
We furthermore performed load testing using locust using the same setup as in the exercises. This yielded an average response time of 130 ms, a median of 100 ms, and a 99th percentile of 1300 ms.


### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did set up the standard monitoring in the google cloud UI. This monitors our cloud runs giving insights into the amount of requests to our api, cpu and memory utilization and potential error occurrences. The monitoring of our cloud buckets also provides insights into the storage data used and how much traffic the storage has. In general it is important to monitor the deployed model to ensure a reliable application and high uptime. Using monitoring one can quickly detect errors and fix them to ensure user satisfaction. It is also useful in regards to google cloud to manage the billing and budgeting depending on cloud usage. We could expand the monitoring system to look at even more specific metrics and set up pipelines for how to handle errors by for example sending emails to relevant stakeholders.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:
We implemented a simple frontend for our API using streamlit, we did this to make the model easily available therefore relying on a simplistic design with only one possible action - uploading an image. This is easily done in streamlit using the file_uploader object, and hereafter calling our backend API. Based on the results we then display the prediction and a chart of the predicted probabilities for the top 10 most likely classes. Furthermore, we used ONNX to save our model to keep it in a standard format avoiding the need of pytorch to load our models. The deployed website can be found here [here](https://frontend-pokedec-228711502156.europe-west3.run.app/).
We have furthermore setup a framework for detecting data drifting.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The biggest challenge was that the github workflows could not be entirely tested locally before submitting them. Therefore there was a lot of trial and error with getting the workflows to run properly. The same goes with building in google cloud resulting in many build submissions and a lot of waiting time. Another pain point on this was to properly parse along the secrets in the Secret Manager the Cloud Run to load an artifact from wandb, as we didn’t know you could attach and expose a secret to a service through the GCP interface.
This was the final solution we landed on after also considering just having the large model in Github, or in a model storage bucket on GCP. The last consideration did also result in the problems with dvc as having two remotes was not easy to handle either.
For the most part the training of the model was okay, though there were some bumps with getting the sweeps and everything setup and running on a new team and also some trial and error on saving artifacts with correct names - here both referring to the torch save and onnx export.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

In general we have in the group been good at keep each other up to date on the progress of the tasks we divided between each other. So while members worked more on some things than others, all are kept continuously kept in the loop.

Student s214611 made the docker containers and different cloudbuild.yaml configurations and made sure that the builds were successfully submitted to google cloud and they were up and running.

Student s214647 setup most of the data processing and training of the model along with sending the sweeps to wandb. He also did profiling for the model, and helped with workflows.

Student s214585 did the evaluation of the trained model on the test set along with adding a workflow for pre-commits. He also worked together with student s214647 on getting the api to work wandb.

Student s201680 implemented different tests including a variety of unit and api tests. He was also a help on other projects.

We have used ChatGPT and GitHub Copilot to help debug our code and write code faster. Especially also helping make comments and doc strings in an aligned manner. Furthermore, ChatGPT has been used in some cases to edit spelling, rephrasing or concatenating pieces of text in some parts of this report.
