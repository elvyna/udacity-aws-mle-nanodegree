## README

Important notes:
- `project-02-model-inference` depends on `sagemaker` SDK, hence, we have to upload the lambda function alongside its dependencies as a `.zip` file. To generate the dependency files, run the following command:

```
pip install --target ./package sagemaker
```

This command will create a `package` folder under that directory. Then, zip all files together:
```
cd package
zip -r ../project-02-model-inference.zip .

cd ..
zip -g project-02-model-inference.zip lambda_function.py
```

Reference: 
- [https://docs.aws.amazon.com/lambda/latest/dg/python-package.html#python-package-create-package-with-dependency](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html#python-package-create-package-with-dependency)