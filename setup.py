from setuptools import setup, find_packages

setup(
    name="aquila-net",
    version="1.0.0",
    description="Physics-Aware Federated GNN for Undersea Cable Anomaly Detection",
    author="AQUILA-NET Team",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch>=2.3.0",
        "torch-geometric>=2.5.3",
        "opacus>=1.4.1",
        "numpy>=1.26.4",
        "scipy>=1.13.0",
        "pandas>=2.2.2",
        "scikit-learn>=1.4.2",
        "plotly>=5.22.0",
        "dash>=2.17.0",
        "dash-bootstrap-components>=1.6.0",
        "requests>=2.31.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.4",
    ],
    extras_require={"dev": ["pytest>=8.2.0"]},
)
