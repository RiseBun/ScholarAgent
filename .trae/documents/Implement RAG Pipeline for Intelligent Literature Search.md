# RAG Pipeline Implementation Plan for ScholarAgent

## Overview

Transform the current keyword-based search engine into an intelligent literature discovery system that mimics LLM thinking through a three-stage RAG pipeline.

## Implementation Steps

### 1. Query Expansion Module (`query_expander.py`)

* Create a new module to rewrite user queries using LLM

* Implement prompt templates for scientific query expansion

* Handle abbreviation expansion (e.g., VLA → Vision-Language-Action)

* Generate synonyms and related terms

* Extract venue filters (e.g., CVPR) from user input

### 2. Enhanced Paper Fetcher (`search_engine.py` modifications)

* Update `PaperFetcher` to accept expanded queries

* Increase retrieval limit to 50-100 papers for broader coverage

* Add venue filtering capabilities

* Implement more robust metadata extraction

### 3. LLM Reranking Module (`content_curator.py`)

* Create a new module for intelligent paper filtering

* Implement relevance scoring using LLM

* Add quality assessment (venue prestige, citation count)

* Generate human-readable explanations for paper selection

### 4. Integration with Streamlit App

* Update UI to show search progress through each pipeline stage

* Display expanded query and reasoning to users

* Present reranked results with relevance scores

* Include paper selection explanations

## Technical Requirements

* LLM API integration (e.g., OpenAI, Claude, or local LLM)

* Enhanced error handling for API calls

* Progress tracking for user feedback

* Efficient batch processing for LLM reranking

## Expected Outcomes

* Significantly improved paper relevance compared to keyword matching

* Ability to find papers with different terminology variations

* Intelligent venue-specific filtering

* Human-readable search reasoning and paper selection explanations

* Performance comparable to ChatGPT/Gemini for literature discovery

## Implementation Strategy

1. Start with query expansion module
2. Enhance paper fetcher capabilities
3. Implement LLM reranking system
4. Integrate with existing Streamlit app
5. Test with sample queries (e.g., "CVPR VLA 自动驾驶")

