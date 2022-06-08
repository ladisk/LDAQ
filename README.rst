SDyPy Template Project
-----------------------

A tempalte to help you start a new projext in the SDyPy ecosystem.


Using the template
------------------

To use this template, you have multiple options. The following two will cover most use cases:

1. You can use GitHub's templating functionality. A new repository will be created on GitHub for your project. Use this option if your project does not yet have an online repository.
   
   Click the "Use this template" button on the project template Github repository (see image below).

    .. image:: images/use_template.png

   Simply select and confirm a name for your new repository, and a copy of this template will be created for you. 

   You can now clone your new repository onto your local machine. If your new repository is located at ``https://github.com/<your_name>/<my_new_project>``, for example:

    .. code-block:: console

        $ git clone https://github.com/<your_name>/<my_new_project>

   A folder named ``<my_new_project>`` will be created on your machine. It is already setup with a connection to your new GitHub repository, and you can begin developing your package!

2. If you already have a repository for your project, located for example at ``https://github.com/<your_name>/<my_existing_project>``, 
   you can use our template by cloning in onto you local machine. This downloads the files into a local folder, with a connection with the online repository already set up.
   Do this by running :

    .. code-block:: console

        $ git clone https://github.com/sdypy/sdypy_template_project

   Our template files will be downloaded into the ``sdypy_template_project`` folder. 
   
   You can now either copy these files into you existing local project folder, or connect the cloned repository in the ``sdypy_template_project`` folder with your existing online repository :

    .. code-block:: console

        $ git remote rm origin
        $ git remote add origin https://github.com/ladisk/<my_existing_project>.git

You are now setup to begin working on your project.

To begin development, install the required packages with :

.. code-block:: console

    $ python -m pip install -r requirements.dev.txt

Now you can replace the core source code modules in ``sdypy_template_project/`` with your code.

Remember to also replace the poject name (``sdypy_template_project``) with your own project name in the following files:

- setup.py
- README.rst
- CONTRIBUTING.rst
- the "sdypy_template_project" directory name

Consider adding unit-tests for your project by modifying the files, found in ``tests/``. The provided test file structure is setup to work with `pytest <https://docs.pytest.org/en/latest/>`_.

To also use the sphinx documentation, modify files in ``docs/source``, or remove the ``docs/`` folder and quickstart a fresh documentation version using the ``sphinx-quickstart`` command (see `Sphinx - Getting started <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_ for more info).


File structure
--------------

The project code is structured as follows:

setup.py
    the Python setup script, used to package the project

requirements.txt
    a list of packages, required to use this project
    
requirements.dev.txt
    a list of packages, required to develop this project

README.rst
    the main projecdt description / documentation file

CONTRIBUTING.rst
    a document containing information for potential contrubutors (developers) of the package

License
    the project License

.travis.yml
    contains the set of instructions to run wit the `TravisCI <https://travis-ci.org/>`_ continuous integration service after the file repository has been updated

.gitignore
    defines the files in the project directory to be excluded from version control

tests/
    contains project unit-tests

sdypy_template_project/
    contains the core project source code, separated into meaningful sub-modules

examples/
    scripts, notebooks with examples to showcase the project

docs/
    the documentation source and built files


(For a more complex and custumuzable project structure, see the `Cookiecutter project <https://github.com/audreyr/cookiecutter-pypackage>`_.)


Building the documentation
--------------------------

By setting up `ReadTheDocs <https://readthedocs.org/>`_, your project documentation can automatically be built and puclished as a publicly available website.

To test your documentation locally, run the following (starting from the main project directory) :

.. code-block:: console

    $ cd docs
    $ make clean
    $ make html

Your documentation files will be built inside the ``docs/build/html`` folder.


Publishing the project
----------------------

You can build your project and publish it to the `Python Package Index <https://pypi.org/>`_ with the following basic steps:

1. Build you project source code :

.. code-block:: console

    $ python setup.py sdist bdist_wheel

The built project can be tested locally by installing the resulting ``.whl`` file, found in the ``dist/`` folder  in a new virtual environemtn:

.. code-block:: console

    $ python -m virtualenv venv
    $ venv/Scripts/activate
    $ python -m pip install <sdypy_template_project-#>.whl 

(replace ``<sdypy_template_project-#>`` above with the actual ``.whl`` file name).

2. Upload the distribution files from ``dist/`` to PyPI :

.. code-block:: console

    $ python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

(``--repository-url https://test.pypi.org/legacy/`` uploads the package to the test PyPI for testing. To publish you package to the main PyPI repository, simply ommit this option from the above command.)

For more information on the publishng process, see this simpel `Python packaging tutorial <https://packaging.python.org/tutorials/packaging-projects/>`_.

3. After that,  the sdypy_template_project will be available on PyPI and can be installed with `pip <https://pip.pypa.io>`_.

.. code-block:: console

    $ pip install sdypy_template_project

After installing sdypy_template_project you can use it like any other Python module.

Here is a simple example with the current example code:

.. code-block:: python

    import sdypy_template_project as iep
    import numpy as np
    import matplotlib.pyplot as plt

    video = np.load('examples/speckle.npy', mmap_mode='r')
    results = iep.get_displacements(video, point=[5, 5], roi_size=[7, 7])

    plt.figure()
    plt.plot(results[0], label='x [px]')
    plt.plot(results[1], label='y [px]')
    plt.legend()
    plt.show()

You can also run this basic example by running the following command in the project base direcotry:

.. code-block:: console

    $ python -m examples.basic_example

The `Read the Docs page <http://sdypy_template_project.readthedocs.io>`_ provides the project documentation.
