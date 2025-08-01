from setuptools import setup, find_packages

setup(
    name="malaria_emulator",
    version="0.1.0",
    description="Streamlit app to estimate malaria incidence and EIR",
    author="Olatunde Ibrahim",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "pandas",
        "matplotlib",
        "seaborn",
        "streamlit",
        "torch",
    ],
    entry_points={
        "console_scripts": [
            "run-emulator=emulator:main",
        ],
    },
)
