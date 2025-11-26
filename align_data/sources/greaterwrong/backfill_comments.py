#!/usr/bin/env python
"""
One-time backfill script for fetching comments for existing GreaterWrong articles.

Usage:
    python -m align_data.sources.greaterwrong.backfill_comments [--source SOURCE] [--dry-run] [--batch-size N]
"""

import argparse
import json
import logging
import re
import sys
import time
from typing import Dict, List, Optional, Set

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from align_data.db.models import Article, ArticleComment
from align_data.db.session import make_session
from align_data.sources.greaterwrong import GREATERWRONG_REGISTRY

logger = logging.getLogger(__name__)

GREATERWRONG_SOURCES = ["lesswrong", "alignmentforum", "eaforum"]


def get_greaterwrong_instance(source_name: str):
    """Get the GreaterWrong instance for a specific source."""
    for instance in GREATERWRONG_REGISTRY:
        if instance.name == source_name:
            return instance
    raise ValueError(f"Unknown source: {source_name}")


def extract_post_id_from_url(url: Optional[str]) -> Optional[str]:
    """Extract post_id from GreaterWrong URL.

    URLs look like:
    - https://www.lesswrong.com/posts/ABC123/post-title
    - https://www.alignmentforum.org/posts/ABC123/post-title
    - https://forum.effectivealtruism.org/posts/ABC123/post-title
    """
    if not url:
        return None

    match = re.search(r'/posts/([a-zA-Z0-9]+)/', url)
    if match:
        return match.group(1)
    return None


def get_post_id_from_article(article: Article) -> Optional[str]:
    """Extract post_id from article metadata or URL."""
    # First try metadata
    meta = article.meta
    if meta:
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = None

        if isinstance(meta, dict) and meta.get("post_id"):
            return meta.get("post_id")

    # Fallback to extracting from URL
    return extract_post_id_from_url(article.url)


def backfill_comments_for_source(
    source_name: str,
    dry_run: bool = False,
    batch_size: int = 100,
    resume_from_id: Optional[int] = None,
) -> Dict[str, int]:
    """
    Backfill comments for all articles from a specific GreaterWrong source.

    Args:
        source_name: The source to backfill (lesswrong, alignmentforum, eaforum)
        dry_run: If True, don't actually write to the database
        batch_size: Number of comments to batch before committing
        resume_from_id: Article _id to resume from (for restarts)

    Returns:
        Dict with statistics
    """
    stats = {
        "articles_processed": 0,
        "articles_skipped_no_post_id": 0,
        "comments_fetched": 0,
        "comments_inserted": 0,
        "comments_skipped_duplicate": 0,
        "errors": 0,
    }

    # Get the GreaterWrong instance for API access
    gw_instance = get_greaterwrong_instance(source_name)

    with make_session() as session:
        # Load all existing comment_ids to avoid duplicates
        existing_comment_ids: Set[str] = set(
            session.scalars(
                select(ArticleComment.comment_id).where(
                    ArticleComment.source == source_name
                )
            ).all()
        )
        logger.info(
            f"Found {len(existing_comment_ids)} existing comments for {source_name}"
        )

        # Query all articles for this source
        query = (
            session.query(Article)
            .filter(Article.source == source_name)
            .order_by(Article._id)
        )

        if resume_from_id:
            query = query.filter(Article._id >= resume_from_id)

        articles = query.all()
        total_articles = len(articles)
        logger.info(f"Found {total_articles} articles to process for {source_name}")

        new_comments_batch: List[ArticleComment] = []

        for idx, article in enumerate(articles):
            article_pk = article._id
            post_id = get_post_id_from_article(article)

            if not post_id:
                stats["articles_skipped_no_post_id"] += 1
                logger.debug(f"Article {article_pk} has no post_id, skipping")
                continue

            title_preview = (
                article.title[:50] + "..." if article.title and len(article.title) > 50 else article.title or "Untitled"
            )
            logger.info(
                f"Processing article {idx + 1}/{total_articles} (id={article_pk}): {title_preview}"
            )

            try:
                # Fetch all comments for this post (handles pagination internally)
                raw_comments = gw_instance.fetch_comments_for_post(post_id)
                stats["comments_fetched"] += len(raw_comments)

                for comment in raw_comments:
                    comment_id = comment.get("_id")
                    if not comment_id:
                        continue

                    # Skip if already exists
                    if comment_id in existing_comment_ids:
                        stats["comments_skipped_duplicate"] += 1
                        continue

                    # Extract comment text
                    text = gw_instance._comment_text(comment)
                    if not text:
                        continue

                    # Extract author
                    user = comment.get("user")
                    author = user.get("displayName") if isinstance(user, dict) else None

                    # Create ArticleComment with only FK (no relationship)
                    comment_model = ArticleComment(
                        comment_id=comment_id,
                        article_id=article_pk,
                        source=source_name,
                        text=text,
                        author=author,
                        posted_at=gw_instance._get_published_date(
                            comment.get("postedAt")
                        ),
                        parent_comment_id=comment.get("parentCommentId"),
                        url=comment.get("pageUrl"),
                    )

                    new_comments_batch.append(comment_model)
                    existing_comment_ids.add(comment_id)
                    stats["comments_inserted"] += 1

                stats["articles_processed"] += 1

                # Commit batch periodically
                if len(new_comments_batch) >= batch_size and not dry_run:
                    try:
                        session.add_all(new_comments_batch)
                        session.commit()
                        logger.info(f"Committed batch of {len(new_comments_batch)} comments")
                    except IntegrityError as exc:
                        logger.warning(f"Integrity error on batch commit, retrying individually: {exc}")
                        session.rollback()
                        for comment in new_comments_batch:
                            try:
                                session.add(comment)
                                session.commit()
                            except IntegrityError:
                                logger.debug(f"Skipping duplicate comment {comment.comment_id}")
                                session.rollback()
                                stats["comments_inserted"] -= 1
                                stats["comments_skipped_duplicate"] += 1
                    new_comments_batch = []

                # Respect API rate limits
                time.sleep(gw_instance.COOLDOWN)

            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Error processing article {article_pk} ({article.url}): {e}")
                continue

        # Final commit for remaining comments
        if new_comments_batch and not dry_run:
            try:
                session.add_all(new_comments_batch)
                session.commit()
                logger.info(f"Final commit of {len(new_comments_batch)} comments")
            except IntegrityError as exc:
                logger.warning(f"Integrity error on final commit, retrying individually: {exc}")
                session.rollback()
                for comment in new_comments_batch:
                    try:
                        session.add(comment)
                        session.commit()
                    except IntegrityError:
                        logger.debug(f"Skipping duplicate comment {comment.comment_id}")
                        session.rollback()
                        stats["comments_inserted"] -= 1
                        stats["comments_skipped_duplicate"] += 1

    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill comments for existing GreaterWrong articles"
    )
    parser.add_argument(
        "--source",
        choices=GREATERWRONG_SOURCES + ["all"],
        default="all",
        help="Source to backfill (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing to database",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of comments to batch before committing (default: 100)",
    )
    parser.add_argument(
        "--resume-from-id",
        type=int,
        help="Article _id to resume from (for restarting interrupted runs)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    sources = GREATERWRONG_SOURCES if args.source == "all" else [args.source]

    if args.dry_run:
        logger.info("DRY RUN MODE - no changes will be written to database")

    total_stats = {
        "articles_processed": 0,
        "articles_skipped_no_post_id": 0,
        "comments_fetched": 0,
        "comments_inserted": 0,
        "comments_skipped_duplicate": 0,
        "errors": 0,
    }

    for source in sources:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Starting backfill for {source}")
        logger.info(f"{'=' * 60}")

        stats = backfill_comments_for_source(
            source_name=source,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            resume_from_id=args.resume_from_id,
        )

        # Print stats for this source
        logger.info(f"\nStats for {source}:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # Aggregate
        for key, value in stats.items():
            total_stats[key] += value

    # Print total stats
    logger.info(f"\n{'=' * 60}")
    logger.info("TOTAL STATS:")
    logger.info(f"{'=' * 60}")
    for key, value in total_stats.items():
        logger.info(f"  {key}: {value}")

    sys.exit(0 if total_stats["errors"] == 0 else 1)


if __name__ == "__main__":
    main()
