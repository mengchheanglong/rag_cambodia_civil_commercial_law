"""
Shared test fixtures and configuration.
"""

import pytest


@pytest.fixture
def sample_article_text_en() -> str:
    """Sample English legal text with articles for testing."""
    return """
BOOK 4 OBLIGATIONS

CHAPTER 1 GENERAL PROVISIONS

Article 1. This Book shall govern obligations arising from contracts,
torts, unjust enrichment, and management of affairs.

Article 2. An obligation is a legal relationship between specific
persons under which one party (the obligor) owes the other party
(the obligee) a specific performance.

CHAPTER 2 FORMATION OF CONTRACTS

Article 315. A contract shall be formed when an offer and an
acceptance thereof are made between the parties and their contents
are in agreement.

Article 316. An offer shall be a manifestation of intention to enter
into a contract made to another party, which contains the essential
elements of the contract with sufficient definiteness.
"""


@pytest.fixture
def sample_article_text_kh() -> str:
    """Sample Khmer legal text with articles for testing."""
    return """
គន្ថី ៤ កាតព្វកិច្ច

ជំពូក ១ បទប្បញ្ញត្តិទូទៅ

មាត្រា ១។ គន្ថីនេះ គ្រប់គ្រងកាតព្វកិច្ចដែលកើតឡើងពីកិច្ចសន្យា

មាត្រា ២។ កាតព្វកិច្ចគឺជាទំនាក់ទំនងផ្លូវច្បាប់រវាងបុគ្គលជាក់លាក់
"""
