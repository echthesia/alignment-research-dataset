"""add article comments table

Revision ID: d1bfe8f5c4e7
Revises: 7d7aae5b6d1a
Create Date: 2025-02-09 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision = "d1bfe8f5c4e7"
down_revision = "7d7aae5b6d1a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "article_comments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("comment_id", sa.String(length=256), nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=256), nullable=True),
        sa.Column("text", mysql.LONGTEXT(), nullable=False),
        sa.Column("author", sa.String(length=256), nullable=True),
        sa.Column("posted_at", sa.DateTime(), nullable=True),
        sa.Column("parent_comment_id", sa.String(length=256), nullable=True),
        sa.Column("url", sa.String(length=1028), nullable=True),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("comment_id"),
    )


def downgrade() -> None:
    op.drop_table("article_comments")
