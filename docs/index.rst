TestGen Copilot Documentation
==============================

**AI-powered test generation and security analysis for Python projects**

TestGen Copilot is a comprehensive CLI tool and VS Code extension that leverages artificial intelligence to automatically generate comprehensive unit tests and highlight potential security vulnerabilities in your Python codebase.

.. image:: https://img.shields.io/pypi/v/testgen-copilot.svg
   :target: https://pypi.org/project/testgen-copilot/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/testgen-copilot.svg
   :target: https://pypi.org/project/testgen-copilot/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/terragonlabs/testgen-copilot.svg
   :target: https://github.com/terragonlabs/testgen-copilot/blob/main/LICENSE
   :alt: License

Quick Start
-----------

Installation
~~~~~~~~~~~~

Install TestGen Copilot using pip:

.. code-block:: bash

   pip install testgen-copilot

Basic Usage
~~~~~~~~~~~

Generate tests for your entire project:

.. code-block:: bash

   testgen generate --project /path/to/your/project

Generate tests for a specific file:

.. code-block:: bash

   testgen generate --file /path/to/your/file.py

Run security analysis:

.. code-block:: bash

   testgen analyze --project /path/to/your/project --security-scan

Key Features
------------

🧪 **Intelligent Test Generation**
   Generate comprehensive unit tests using advanced AI models (GPT-4, Claude-3)

🛡️ **Security Analysis**
   Multi-layer security scanning with OWASP compliance checking

📊 **Coverage Analysis**
   Track test coverage and identify gaps in your test suite

🚀 **Autonomous Execution**
   Automated backlog management with DORA metrics tracking

🔧 **IDE Integration**
   Native VS Code extension with real-time suggestions

🐳 **Container Support**
   Docker integration with security scanning capabilities

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   configuration
   cli-reference
   vscode-extension

.. toctree::
   :maxdepth: 2
   :caption: Features

   test-generation
   security-analysis
   coverage-analysis
   autonomous-execution

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules
   api/testgen_copilot

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   architecture
   development-setup
   release-process

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   changelog
   roadmap
   faq
   troubleshooting

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`